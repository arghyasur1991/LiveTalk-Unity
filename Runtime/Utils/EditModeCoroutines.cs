#if UNITY_EDITOR
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace LiveTalk.Utils
{
    /// <summary>
    /// Runs LiveTalk's producer coroutines in edit mode without the player
    /// loop.
    ///
    /// Unity services the player loop — and with it every <c>MonoBehaviour</c>
    /// coroutine — only while the editor is the foreground application.
    /// <c>EditorApplication.QueuePlayerLoopUpdate</c> does not change that
    /// (measured 2026-09-22: a host coroutine advanced 0 times in 177 s
    /// unfocused, 10 times in 350 ms once focused). <c>EditorApplication.update</c>
    /// keeps ticking unfocused, so an edit-mode bake steps its iterators from
    /// here instead of <c>StartCoroutine</c>.
    ///
    /// Each update steps every routine until it blocks: a
    /// <see cref="CustomYieldInstruction"/> still waiting, an
    /// <see cref="AsyncOperation"/> not done, or two consecutive <c>null</c>
    /// yields (the shape of <see cref="TaskYield"/> polling a task that has
    /// not settled). Work between yields runs in one tick; a wait costs at
    /// most one editor update. A time budget keeps the editor responsive.
    /// </summary>
    internal static class EditModeCoroutines
    {
        const double BudgetSeconds = 0.25;

        sealed class Routine
        {
            public readonly Stack<IEnumerator> Stack = new();
            public object Pending;
        }

        static readonly List<Routine> Routines = new();
        static bool _hooked;

        /// <summary>Number of routines currently being driven.</summary>
        internal static int Active => Routines.Count;

        internal static void Start(IEnumerator routine)
        {
            if (routine == null)
                throw new ArgumentNullException(nameof(routine));
            var r = new Routine();
            r.Stack.Push(routine);
            Routines.Add(r);
            if (!_hooked)
            {
                EditorApplication.update += Tick;
                _hooked = true;
            }
        }

        /// <summary>Drops every routine without running their remaining code.</summary>
        internal static void StopAll()
        {
            foreach (var r in Routines)
                DisposeStack(r.Stack);
            Routines.Clear();
        }

        static void Tick()
        {
            if (Routines.Count == 0)
            {
                EditorApplication.update -= Tick;
                _hooked = false;
                return;
            }

            double deadline = EditorApplication.timeSinceStartup + BudgetSeconds;
            for (int i = Routines.Count - 1; i >= 0; i--)
            {
                var r = Routines[i];
                bool finished;
                try
                {
                    finished = Step(r, deadline);
                }
                catch (Exception e)
                {
                    // Producers run under TaskYield.Guard, so this is a fault in
                    // the guard itself or in a bare routine. Log and drop it.
                    Debug.LogException(e);
                    DisposeStack(r.Stack);
                    finished = true;
                }
                if (finished)
                    Routines.RemoveAt(i);
            }
        }

        /// <summary>Returns true when the routine has completed.</summary>
        static bool Step(Routine r, double deadline)
        {
            int nullsInARow = 0;
            while (r.Stack.Count > 0)
            {
                if (!IsReady(r.Pending))
                    return false;
                r.Pending = null;

                var e = r.Stack.Peek();
                if (!e.MoveNext())
                {
                    r.Stack.Pop();
                    continue;
                }

                object cur = e.Current;
                if (cur is IEnumerator nested)
                {
                    r.Stack.Push(nested);
                    nullsInARow = 0;
                    continue;
                }
                if (cur == null)
                {
                    // TaskYield.Wait polls with null while a task is running.
                    // One null is a plain frame break; a second in the same
                    // tick means nothing progressed, so hand the tick back.
                    if (++nullsInARow >= 2)
                        return false;
                }
                else
                {
                    r.Pending = cur;
                    if (!IsReady(cur))
                        return false;
                    nullsInARow = 0;
                }

                if (EditorApplication.timeSinceStartup > deadline)
                    return false;
            }
            return true;
        }

        static bool IsReady(object yielded)
        {
            switch (yielded)
            {
                case null:
                    return true;
                case CustomYieldInstruction custom:
                    return !custom.keepWaiting;
                case AsyncOperation op:
                    return op.isDone;
                default:
                    // WaitForSeconds and other YieldInstructions have no
                    // edit-mode clock; treat them as a one-tick break.
                    return true;
            }
        }

        static void DisposeStack(Stack<IEnumerator> stack)
        {
            while (stack.Count > 0)
            {
                if (stack.Pop() is IDisposable d)
                {
                    try { d.Dispose(); }
                    catch (Exception e) { Debug.LogException(e); }
                }
            }
        }
    }
}
#endif
