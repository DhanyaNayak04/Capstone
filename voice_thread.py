# voice_thread.py

import queue
import json
import re
import difflib
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from state import command_queue, tts_lock
try:
    # For runtime gating of which commands are allowed while reading/tts is active
    from reading_manager import reading_status
except Exception:
    reading_status = None

ALLOWED_COMMANDS = [
    "stop", "start", "exit", "quit",
    "pause", "pause reading", "resume reading", "continue reading", "stop reading",
    "what is in front of me", "what's in front of me",
    "what is infront of me", "what's infront of me",
    "who is in front of me", "who's in front of me",
    "who is infront of me", "who's infront of me"
]

# We'll treat read commands specially (they aren't in ALLOWED_COMMANDS because
# they need to be handed to the OCR pipeline). Keep variants here for matching.
READ_COMMAND_PHRASES = [
    "read this", "read that", "read the text", "read text", "read"
]

# Control phrases for reading session (exact match only, to avoid false positives like 'pause rating')
READ_CONTROL_PHRASES = {
    "pause": [
        "pause",
        "pause reading",
        "pause the reading",
    ],
    "resume reading": [
        "resume",
        "resume reading",
        "resume the reading",
        "continue",
        "continue reading",
        "continue the reading",
    ],
    "stop reading": [
        "stop reading",
        "cancel reading",
        "end reading",
    ],
}

# Build limited grammar list for VOSK keyword/grammar mode.
# Keep everything lowercase to match our _normalize_text recognition.
_GRAMMAR_SET = set()
_GRAMMAR_SET.update(ALLOWED_COMMANDS)
_GRAMMAR_SET.update(READ_COMMAND_PHRASES)
for _vlist in READ_CONTROL_PHRASES.values():
    _GRAMMAR_SET.update(_vlist)
VOCAB_GRAMMAR = sorted(_GRAMMAR_SET)


def _normalize_text(s: str) -> str:
    s = s.lower()
    s = s.replace("what's", "what is")
    s = s.replace("whats", "what is")
    s = s.replace("who's", "who is")
    s = s.replace("whos", "who is")
    s = s.replace("infront", "in front")
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

_NORMALIZED_TO_COMMAND = {_normalize_text(cmd): cmd for cmd in ALLOWED_COMMANDS}
_NORMALIZED_COMMANDS = list(_NORMALIZED_TO_COMMAND.keys())


# --- MODIFIED _match_command FUNCTION ---
def _match_command(recognized_text: str, cutoff: float = 0.7):
    """
    Try to match recognized_text to one of ALLOWED_COMMANDS.
    This version prioritizes an exact match before falling back to fuzzy matching.
    """
    norm = _normalize_text(recognized_text)

    # Reading control commands first (exact match only)
    for canonical, variants in READ_CONTROL_PHRASES.items():
        for v in variants:
            if norm == v:
                return canonical

    # Check read phrases (exact)
    for phrase in READ_COMMAND_PHRASES:
        if norm == phrase:
            return phrase

    # 1. (NEW) Prioritize an exact match on the normalized text.
    if norm in _NORMALIZED_TO_COMMAND:
        return _NORMALIZED_TO_COMMAND[norm]

    # 2. Check for short, single-word commands.
    for short in ("stop", "start", "exit", "quit"):
        if norm == short:
            return short

    # 3. (FALLBACK) Disable broad fuzzy match to avoid false positives like 'cause' -> 'pause'.
    # If needed, you can re-enable fuzzy only for long phrases (>= 10 chars) and exclude single-word commands.
    # Example:
    # if len(norm) >= 10:
    #     long_cmds = [c for c in _NORMALIZED_COMMANDS if len(c) >= 10]
    #     matches = difflib.get_close_matches(norm, long_cmds, n=1, cutoff=0.85)
    #     if matches:
    #         return _NORMALIZED_TO_COMMAND[matches[0]]

    return None


def voice_command_thread():
    # This function remains unchanged.
    model = Model(lang="en-us")
    # Limited grammar: recognizer will strongly bias toward the given phrases
    import json as _json
    recognizer = KaldiRecognizer(model, 16000, _json.dumps(VOCAB_GRAMMAR))
    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(f"[VOSK] Status: {status}")
        # Always capture mic input so we can recognize control commands while TTS is speaking
        q.put(bytes(indata))

    print("[VOSK] Voice command thread started.")
    # Smaller blocksize reduces command latency (approx blocksize/samplerate seconds)
    with sd.RawInputStream(samplerate=16000, blocksize=1600, dtype='int16', channels=1, callback=callback):
        while True:
            try:
                data = q.get()
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "")
                    if text:
                        print(f"[VOSK] Recognized: '{text}'")
                        matched_command = _match_command(text)
                        if matched_command:
                            # Gate commands during active reading or TTS playback to reduce false triggers
                            allow = True
                            active = False
                            paused = False
                            if reading_status:
                                try:
                                    st = reading_status()
                                    active = bool(st.get('active'))
                                    paused = bool(st.get('paused'))
                                except Exception:
                                    pass
                            if tts_lock.locked() or active:
                                # Only allow reading controls while TTS or reading is active
                                if matched_command not in ("pause", "pause reading", "resume reading", "continue reading", "stop reading"):
                                    allow = False
                            if allow:
                                print(f"[VOSK] Command matched: '{matched_command}'")
                                command_queue.put(matched_command)
                            else:
                                print(f"[VOSK] Ignored command during playback: '{matched_command}'")
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[VOSK] Error: {e}")