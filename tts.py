"""Text-to-speech queue handling."""
from gtts import gTTS
# Prefer playsound3 (more robust, modern backends on Windows) and fall back to playsound
try:
    from playsound3 import playsound
    _PLAYSOUND_BACKEND = "playsound3"
except Exception:
    try:
        from playsound import playsound  # older playsound package
        _PLAYSOUND_BACKEND = "playsound"
    except Exception:
        raise ImportError("No supported playsound package found. Install 'playsound3' or 'playsound'.")
import tempfile, os
import heapq
from state import tts_queue, tts_queue_lock, tts_queue_event, tts_lock


def tts_worker():
    """Background worker that consumes the priority TTS queue."""
    print(f"[TTS] Worker started (gTTS mode). using {_PLAYSOUND_BACKEND}")
    while True:
        tts_queue_event.wait()
        with tts_queue_lock:
            if not tts_queue:
                tts_queue_event.clear()
                continue
            priority, text = heapq.heappop(tts_queue)
            if text is None:
                break
        print(f"[TTS] Speaking: {text}")
        with tts_lock:
            try:
                tts = gTTS(text=text, lang='en')
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                    tts.save(fp.name)
                    temp_path = fp.name
                try:
                    playsound(temp_path)
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            except Exception as e:
                print(f"[TTS] Error: {e}")


def speak(text: str, priority: int = 5):
    """Queue a phrase to be spoken. Lower priority value = higher priority."""
    print(f"[SPEAK] Queued: {text} (priority {priority})")
    with tts_queue_lock:
        heapq.heappush(tts_queue, (priority, text))
        tts_queue_event.set()