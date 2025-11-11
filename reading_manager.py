"""Incremental OCR text reading with pause/resume/stop control.

This module manages a single active reading session that can be:
- started with a text string
- paused immediately (stopping current audio playback)
- resumed from the exact character offset
- stopped (session torn down and other commands allowed)

It uses gTTS for synthesis per chunk and pygame.mixer for playback so we can
interrupt mid-chunk.
"""
from __future__ import annotations
import threading, tempfile, os, time
from typing import Optional
from gtts import gTTS
from mutagen.mp3 import MP3
import re

# pygame mixer init is lazy; guard import errors gracefully
try:
    import pygame
    pygame.mixer.init()
    _PYGAME_OK = True
except Exception as e:
    print(f"[READ] pygame mixer init failed ({e}); falling back to blocking playsound.")
    _PYGAME_OK = False
    try:
        from playsound3 import playsound as _playsound
    except Exception:
        from playsound import playsound as _playsound

from state import tts_pause_event, tts_stop_event, tts_lock
from tts import speak  # reuse short speak prompts via existing queue

# Session state held in module globals (single session model)
_session_lock = threading.Lock()
_read_thread: Optional[threading.Thread] = None
_current_text: Optional[str] = None
_position: int = 0
_active: bool = False
_paused: bool = False
_last_chunk_start: int = 0
_last_chunk_end: int = 0
_last_chunk_text: str = ""
_last_chunk_duration: float = 0.0
_last_chunk_played_ratio: float = 0.0  # fraction of current chunk played (0..1)

_CHUNK_MAX = 240  # balance responsiveness with fewer boundaries

_SENTENCE_END_RE = re.compile(r"([.!?])\s+")


def _next_chunk(text: str, start: int) -> tuple[str, int]:
    """Return next chunk and new position.
    Prefer sentence boundaries; fall back to fixed size.
    """
    remaining = text[start:]
    if not remaining:
        return "", start
    if len(remaining) <= _CHUNK_MAX:
        return remaining, len(text)
    # Find sentence end before limit
    snippet = remaining[:_CHUNK_MAX]
    match_iter = list(_SENTENCE_END_RE.finditer(snippet))
    if match_iter:
        last = match_iter[-1]
        end = start + last.end()
        return text[start:end], end
    # No sentence boundary; cut at max
    end = start + _CHUNK_MAX
    return text[start:end], end


def _play_audio(path: str):
    # Ensure exclusive playback so TTS queue and reading chunks don't overlap voices
    with tts_lock:
        if _PYGAME_OK:
            try:
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                start_time = time.time()
                while pygame.mixer.music.get_busy():
                    if tts_stop_event.is_set():
                        pygame.mixer.music.stop()
                        break
                    if tts_pause_event.is_set():
                        # let the reader loop handle resume; break now
                        pygame.mixer.music.pause()
                        break
                    time.sleep(0.01)
                # Record approximate played ratio using duration when available
                try:
                    elapsed = max(0.0, time.time() - start_time)
                    if _last_chunk_duration > 0:
                        ratio = min(1.0, elapsed / _last_chunk_duration)
                        with _session_lock:
                            global _last_chunk_played_ratio
                            _last_chunk_played_ratio = ratio
                except Exception:
                    pass
            except Exception as e:
                print(f"[READ] pygame playback error: {e}")
        else:
            # Blocking playback; can't interrupt mid-chunk so keep chunks small
            _playsound(path)


def _synthesize(text: str) -> str:
    tts = gTTS(text=text, lang='en')
    fp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    temp_path = fp.name
    fp.close()
    tts.save(temp_path)
    return temp_path


def _cleanup_temp(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _reader_loop():
    global _position, _active, _paused, _current_text
    print("[READ] Reader thread started")
    while True:
        with _session_lock:
            local_active = _active
            text = _current_text
            pos = _position
        if not local_active:
            print("[READ] Reader thread terminating")
            return
        if text is None or pos >= len(text):
            # Graceful finish without double announcements
            stop_reading(finished=True)
            return
        if tts_stop_event.is_set():
            stop_reading()
            return
        if tts_pause_event.is_set():
            time.sleep(0.05)
            continue
        chunk, new_pos = _next_chunk(text, pos)
        # remember bounds of this chunk so we can replay it if pause occurs mid-chunk
        with _session_lock:
            global _last_chunk_start, _last_chunk_end, _last_chunk_text, _last_chunk_played_ratio, _last_chunk_duration
            _last_chunk_start, _last_chunk_end = pos, new_pos
            _last_chunk_text = chunk
            _last_chunk_played_ratio = 0.0
            _last_chunk_duration = 0.0
        if not chunk:
            stop_reading()
            return
        print(f"[READ] Chunk {pos}->{new_pos} ({len(chunk)} chars)")
        # synthesize
        path = _synthesize(chunk)
        # determine duration if possible
        try:
            audio_info = MP3(path)
            with _session_lock:
                _last_chunk_duration = float(getattr(audio_info.info, 'length', 0.0) or 0.0)
        except Exception:
            pass
        _play_audio(path)
        _cleanup_temp(path)
        # Advance position only if the chunk was fully played without pause/stop
        with _session_lock:
            if not tts_stop_event.is_set() and not tts_pause_event.is_set():
                _position = new_pos
            else:
                # On pause, try to resume inside this chunk based on played ratio
                if tts_pause_event.is_set() and _last_chunk_played_ratio > 0.02:
                    offset = int(len(_last_chunk_text) * _last_chunk_played_ratio)
                    _position = min(new_pos, _last_chunk_start + max(0, offset - 2))  # small safety backtrack
                else:
                    _position = _last_chunk_start
        # Short gap to allow voice recognition capture
        time.sleep(0.05)


def start_reading(text: str):
    """Start or restart an incremental reading session."""
    global _read_thread, _current_text, _position, _active, _paused
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        speak("No readable text detected.")
        return
    with _session_lock:
        # If already active with same text and not paused, ignore
        if _active and _current_text == cleaned and not _paused:
            speak("Already reading this text.")
            return
        # If active with different text, stop first
        if _active and _current_text != cleaned:
            speak("Replacing current reading with new text.")
            _active = False
    if _read_thread and _read_thread.is_alive():
        tts_stop_event.set()
        _read_thread.join(timeout=1)
        tts_stop_event.clear()
    with _session_lock:
        _current_text = cleaned
        _position = 0 if not _active else _position  # restart if previous stopped
        _active = True
        _paused = False
    tts_pause_event.clear()
    tts_stop_event.clear()
    # Preview
    preview = cleaned[:160]
    # Start directly at position 0 without speaking a separate preview to avoid duplication
    speak("Reading text.")
    _read_thread = threading.Thread(target=_reader_loop, daemon=True)
    _read_thread.start()


def pause_reading():
    global _paused
    with _session_lock:
        if not _active:
            speak("No active reading to pause.")
            return
        if _paused:
            speak("Already paused.")
            return
        _paused = True
    tts_pause_event.set()
    if _PYGAME_OK:
        try:
            # stop current audio so reader can promptly idle and replay same chunk on resume
            pygame.mixer.music.stop()
        except Exception:
            pass
    # Avoid speaking here to prevent TTS feedback triggering mis-recognition


def resume_reading():
    global _paused
    with _session_lock:
        if not _active:
            speak("No active reading session.")
            return
        if not _paused:
            speak("Reading is not paused.")
            return
        _paused = False
    tts_pause_event.clear()
    if _PYGAME_OK:
        try:
            pygame.mixer.music.unpause()
        except Exception:
            pass
    # Avoid speaking here to prevent TTS feedback triggering mis-recognition


def stop_reading(finished: bool = False):
    """Stop the active reading session.

    finished=True indicates natural end (announces only Finished reading.)
    finished=False indicates user interruption (announces Stopped reading.)
    """
    global _active, _paused, _current_text, _read_thread
    with _session_lock:
        if not _active:
            if not finished:
                speak("No active reading.")
            return
        _active = False
        _paused = False
    tts_stop_event.set()
    if _PYGAME_OK:
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass
    # Avoid joining from same thread
    if _read_thread and _read_thread.is_alive() and threading.current_thread() is not _read_thread:
        _read_thread.join(timeout=1)
    tts_stop_event.clear()
    tts_pause_event.clear()
    with _session_lock:
        _current_text = None
    if finished:
        # Silent finish to avoid TTS feedback; UI/state reflects inactive
        pass


def reading_status() -> dict:
    with _session_lock:
        return {
            'active': _active,
            'paused': _paused,
            'position': _position,
            'total': len(_current_text) if _current_text else 0,
        }
