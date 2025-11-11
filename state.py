"""Shared mutable state and synchronization primitives."""
import threading, queue

# Locks and events
tts_lock = threading.Lock()

# TTS control events (for pause/resume/stop incremental reading playback in reading_manager)
# These do NOT affect the standard queued speak() messages; they are used only by
# the live reading session so voice recognition can still capture commands.
tts_pause_event = threading.Event()  # set => paused, clear => playing
tts_stop_event = threading.Event()   # set => immediate stop requested

# Queues / priority queue infra
import heapq

tts_queue = []  # list of (priority, text)
tts_queue_lock = threading.Lock()
tts_queue_event = threading.Event()

# Command queue from voice recognition
command_queue = queue.Queue()

# Frame / detection state
latest_frame = [None]  # mutable container for latest frame
frame_lock = threading.Lock()

latest_objects = set()
# Store latest detected bounding boxes as list of dicts: {'label': str, 'xyxy': (x1,y1,x2,y2), 'area': float}
latest_boxes = []
# Make obstacle_detected a mutable container so threads can update it in-place
# (assigning a bare bool in another module won't update the object imported from this module).
obstacle_detected = [False]
context_lock = threading.Lock()

# Detection active flag (mutable container so threads see updates)
detection_active = [False]

# Obstacle warning bookkeeping
obstacle_warning_state = {
    'active': False,
    'count': 0,
}
