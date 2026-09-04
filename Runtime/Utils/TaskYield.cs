using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.ExceptionServices;
using System.Threading.Tasks;
using UnityEngine;

namespace LiveTalk.Utils
{
    /// <summary>
    /// Task → coroutine bridge.
    ///
    /// Coroutines have no exception channel, so the pipeline consumes Tasks by
    /// yielding until they finish and then reading the result. The tempting
    /// test for "finished" is <see cref="Task.IsCompleted"/>, but that is true
    /// for a <b>faulted</b> and a <b>cancelled</b> task as well as a successful
    /// one. Code that yields on <c>IsCompleted</c> and then reads
    /// <c>task.Result</c> either throws an <see cref="AggregateException"/>
    /// from the property getter or, when the result is not needed, silently
    /// carries on as if the step had succeeded. The failure then surfaces
    /// later as a misleading error ("Model is not initialized", "Character
    /// voice not loaded", "Generated audio clip is null") or not at all — a
    /// producer that never marks its stream finished leaves every consumer
    /// waiting forever.
    ///
    /// <see cref="Wait(Task, string)"/> and <see cref="Wait{T}(Task{T}, Action{T}, string)"/>
    /// yield until the task settles, then rethrow the original exception
    /// (with its stack) when it faulted. Because the throw happens inside the
    /// coroutine, the iterator's own <c>finally</c> blocks run on the way out:
    /// a producer that wraps its body in <c>try { … } finally { stream.Finished
    /// = true; }</c> is guaranteed to release its consumers, and a lease taken
    /// with <c>AcquireAsync</c> is guaranteed to be released.
    ///
    /// <see cref="Guard"/> is the other half. A public <c>IEnumerator</c> API
    /// with an <c>onError</c> callback runs its body under a guard, which
    /// drives the body (and every coroutine it nests) itself so that any
    /// exception — from a bridged task or a plain <c>throw</c> — lands in one
    /// <c>catch</c>, disposes the suspended iterators still on the stack (which
    /// runs their pending <c>finally</c> blocks), and is routed to
    /// <c>onError</c>. The host's coroutine keeps running; <c>onComplete</c>
    /// is never reached for a half-built result.
    /// </summary>
    internal static class TaskYield
    {
        /// <summary>
        /// Yields until <paramref name="task"/> settles. Rethrows the task's
        /// exception if it faulted or was cancelled; returns normally otherwise.
        /// </summary>
        /// <param name="task">The task to wait for.</param>
        /// <param name="context">Short label for the log line on failure, e.g. <c>"MuseTalkInference.StartGeneratorSession"</c>.</param>
        public static IEnumerator Wait(Task task, string context = null)
        {
            if (task == null)
                throw new ArgumentNullException(nameof(task));

            while (!task.IsCompleted)
                yield return null;

            ThrowIfNotSucceeded(task, context);
        }

        /// <summary>
        /// Yields until <paramref name="task"/> settles. On success invokes
        /// <paramref name="onResult"/> with the result; on fault or
        /// cancellation rethrows and never invokes it.
        /// </summary>
        /// <param name="task">The task to wait for.</param>
        /// <param name="onResult">Receives the result on success only.</param>
        /// <param name="context">Short label for the log line on failure.</param>
        public static IEnumerator Wait<T>(Task<T> task, Action<T> onResult, string context = null)
        {
            if (task == null)
                throw new ArgumentNullException(nameof(task));

            while (!task.IsCompleted)
                yield return null;

            ThrowIfNotSucceeded(task, context);
            onResult?.Invoke(task.Result);
        }

        /// <summary>
        /// Runs <paramref name="body"/> as a coroutine, flattening any nested
        /// <see cref="IEnumerator"/> it yields, so that every <c>MoveNext</c>
        /// executes inside this method's <c>try</c>. An exception thrown by the
        /// body or by anything it nests is logged once, the iterators still
        /// suspended on the stack are disposed (running their <c>finally</c>
        /// blocks), and <paramref name="onError"/> is invoked. The guard then
        /// completes normally, so the host coroutine that yielded on it keeps
        /// running.
        ///
        /// Yield instructions that are not plain iterators (<c>null</c>,
        /// <see cref="YieldInstruction"/>, <see cref="CustomYieldInstruction"/>,
        /// <see cref="AsyncOperation"/>) are handed to Unity unchanged.
        /// </summary>
        /// <param name="body">The coroutine to run.</param>
        /// <param name="onError">Receives the exception and owns reporting it. When null the failure is logged here instead.</param>
        /// <param name="context">Short label for the log line, e.g. <c>"LiveTalkAPI.CreateCharacterAsync"</c>.</param>
        public static IEnumerator Guard(IEnumerator body, Action<Exception> onError, string context = null)
        {
            if (body == null)
                throw new ArgumentNullException(nameof(body));

            var stack = new Stack<IEnumerator>();
            stack.Push(body);

            try
            {
                while (stack.Count > 0)
                {
                    var top = stack.Peek();
                    bool moved;
                    object current = null;
                    Exception failure = null;

                    try
                    {
                        moved = top.MoveNext();
                        if (moved)
                            current = top.Current;
                    }
                    catch (Exception ex)
                    {
                        moved = false;
                        failure = ex;
                    }

                    if (failure != null)
                    {
                        // `top` has already unwound its own finally blocks.
                        // The iterators below it are still suspended at their
                        // yield; Dispose runs their pending finally blocks.
                        stack.Pop();
                        DisposeAll(stack);

                        // A faulted Task has already been logged with its
                        // stack by Wait. With an onError the host owns the
                        // report, as it does for every other onError path in
                        // the API; without one the failure must not vanish.
                        if (onError == null)
                            Logger.LogError($"{Prefix(context)}{failure.GetType().Name}: {failure.Message}");
                        else
                            onError(failure);
                        yield break;
                    }

                    if (!moved)
                    {
                        stack.Pop();
                        continue;
                    }

                    // CustomYieldInstruction implements IEnumerator; Unity
                    // knows how to poll it, so it must not be flattened.
                    if (current is IEnumerator nested && current is not CustomYieldInstruction)
                    {
                        stack.Push(nested);
                        continue;
                    }

                    yield return current;
                }
            }
            finally
            {
                // Reached on normal completion (empty stack, no-op) or when the
                // host stops this coroutine and Unity disposes the iterator.
                DisposeAll(stack);
            }
        }

        private static void ThrowIfNotSucceeded(Task task, string context)
        {
            if (task.IsCanceled)
            {
                var cancelled = new OperationCanceledException("Task was cancelled.");
                Logger.LogError($"{Prefix(context)}Task was cancelled.");
                throw cancelled;
            }

            if (!task.IsFaulted)
                return;

            // GetBaseException unwraps the AggregateException that Task builds
            // around the real failure, so the log and the rethrow both carry
            // the exception the producer actually threw.
            var ex = task.Exception.GetBaseException();
            Logger.LogError($"{Prefix(context)}Task faulted: {ex}");
            ExceptionDispatchInfo.Capture(ex).Throw();
        }

        private static void DisposeAll(Stack<IEnumerator> stack)
        {
            while (stack.Count > 0)
            {
                var e = stack.Pop();
                if (e is not IDisposable d)
                    continue;

                try
                {
                    d.Dispose();
                }
                catch (Exception ex)
                {
                    Logger.LogError($"[TaskYield] finally block threw while unwinding: {ex}");
                }
            }
        }

        private static string Prefix(string context)
            => string.IsNullOrEmpty(context) ? "[LiveTalk] " : $"[{context}] ";
    }
}
