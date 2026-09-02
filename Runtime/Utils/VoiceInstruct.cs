using System.Text;
using LiveTalk.API;

namespace LiveTalk.Utils
{
    /// <summary>
    /// Turns LiveTalk's gender / pitch / speed knobs into the natural-language
    /// instruct Qwen3-TTS VoiceDesign actually takes.
    ///
    /// This lives here, not in the TTS package: which words describe a voice is
    /// a product decision, it is English-only, and an engine has no business
    /// composing prose. Hosts that want full control should pass
    /// <c>voiceInstruct</c> and leave the knobs at their defaults.
    /// </summary>
    internal static class VoiceInstruct
    {
        /// <summary>
        /// Knobs first, then any host-supplied notes. Callers that display a
        /// preview of what the model will hear must use this same method so the
        /// preview and the generate cannot drift.
        /// </summary>
        public static string Compose(Gender gender, Pitch pitch, Speed speed, string notes)
        {
            var sb = new StringBuilder();
            sb.Append(GenderPhrase(gender));
            sb.Append(", ");
            sb.Append(PitchPhrase(pitch));
            sb.Append(" pitch, ");
            sb.Append(SpeedPhrase(speed));
            sb.Append(" speaking rate.");
            if (!string.IsNullOrWhiteSpace(notes))
            {
                sb.Append(' ');
                sb.Append(notes.Trim());
            }
            return sb.ToString();
        }

        public static string GenderPhrase(Gender gender) =>
            gender == Gender.Female ? "Female" : "Male";

        public static string PitchPhrase(Pitch pitch) => pitch switch
        {
            Pitch.VeryLow => "very low",
            Pitch.Low => "low",
            Pitch.High => "high",
            Pitch.VeryHigh => "very high",
            _ => "medium",
        };

        public static string SpeedPhrase(Speed speed) => speed switch
        {
            Speed.VeryLow => "very slow",
            Speed.Low => "slow",
            Speed.High => "fast",
            Speed.VeryHigh => "very fast",
            _ => "medium",
        };
    }
}
