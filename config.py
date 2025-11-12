"""Configuration constants for smart glasses system.

Notes:
- Do NOT hard-code API keys in source. Prefer environment variables.
- For local development on Windows PowerShell, set env vars like:
	$env:GEMINI_API_KEY = "your-key"
	$env:GEMINI_MODEL = "gemini-2.5-flash-preview-09-2025"  # optional
"""

import os
# Load environment variables from a local .env file if present
try:
	from dotenv import load_dotenv  # type: ignore
	load_dotenv()
except Exception:
	# Safe to ignore if python-dotenv isn't installed; env vars from shell still work
	pass


CAMERA_URL = 'http://192.168.0.3:4747/video'

# Gemini API configuration
# Prefer environment variables and fall back to legacy API_KEY if present.
# Supported env vars:
# - GEMINI_API_KEY: Your Google Generative Language API key
# - GEMINI_MODEL:   Model name, defaults to a fast multimodal model

# Backward compatibility: if an older codebase used `API_KEY`, still honor it.
_LEGACY_KEY = os.getenv("API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or _LEGACY_KEY or ""

# Default to a broadly available flash model; can be overridden via env.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-preview-09-2025")

# Optional behavior flags
# If True, we'll return Gemini result directly (no overlay crop heuristic)
OCR_GEMINI_DIRECT = os.getenv("OCR_GEMINI_DIRECT", "1") in ("1", "true", "True")

# Maintain prior attribute for any imports elsewhere (deprecated)
API_KEY = GEMINI_API_KEY
