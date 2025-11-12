"""Clean OCR implementation used by ocr.py shim.

Keep this module self-contained so it can be edited safely.
"""
from typing import List, Optional, Tuple
import os
import re
import cv2
import numpy as np
from PIL import Image
import base64
import requests

try:
    # Local config for API keys and model selection
    from config import GEMINI_API_KEY, GEMINI_MODEL, OCR_GEMINI_DIRECT
except Exception:
    GEMINI_API_KEY = ""
    GEMINI_MODEL = "gemini-2.0-flash"

__all__ = ["read_text_from_frame"]


try:
    import pytesseract
except Exception:
    pytesseract = None


def _ensure_tesseract_on_windows() -> None:
    if pytesseract is None:
        return
    if os.name == 'nt':
        common = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(common):
            try:
                pytesseract.pytesseract.tesseract_cmd = common
            except Exception:
                pass


def _to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def _np_to_jpeg_base64(img_bgr: np.ndarray, max_side: int = 1280) -> Tuple[str, str]:
    """Convert BGR np image to reasonably sized JPEG and base64-encode it.

    Returns (base64_string, mime_type). Scales down the image to reduce latency/cost.
    """
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Empty image")
    h, w = img_bgr.shape[:2]
    scale = 1.0
    m = max(h, w)
    if m > max_side:
        scale = max_side / float(m)
        new_w, new_h = int(w * scale), int(h * scale)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Encode as medium-quality JPEG
    ok, buf = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    b64 = base64.b64encode(buf.tobytes()).decode('ascii')
    return b64, 'image/jpeg'


def _call_gemini_transcribe(img_bgr: np.ndarray, timeout: float = 20.0) -> str:
    """Call Gemini Vision API to transcribe visible text only.

    Uses the same prompt constraint as the image-extractor: return only text seen,
    no commentary, no completions of partial words.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured. Set it via environment variable.")

    b64, mime = _np_to_jpeg_base64(img_bgr)
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

    prompt = (
        "Transcribe all text visible in this image. Do not add any extra information, "
        "comments, or formatting. Do not try to complete any partial sentences. Just "
        "return the exact text you see."
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": mime, "data": b64}},
                ]
            }
        ]
    }

    headers = {"Content-Type": "application/json"}
    r = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"Gemini API error {r.status_code}: {r.text[:512]}")

    data = r.json()
    # Expected path: candidates[0].content.parts[0].text
    try:
        candidates = data.get("candidates") or []
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts and isinstance(parts[0], dict) and "text" in parts[0]:
                return (parts[0]["text"] or "").strip()
    except Exception:
        pass
    raise RuntimeError("Could not parse text from Gemini response")


def _preprocess_crop(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) < 800:
        img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 75, 75)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def _add_border(img: np.ndarray, px: int = 20, color: int = 255) -> np.ndarray:
    """Add a constant border around the image to avoid edge-clipping of last characters."""
    if len(img.shape) == 2:
        return cv2.copyMakeBorder(img, px, px, px, px, cv2.BORDER_CONSTANT, value=color)
    else:
        return cv2.copyMakeBorder(img, px, px, px, px, cv2.BORDER_CONSTANT, value=(color, color, color))


def _preprocess_variants(img: np.ndarray) -> List[np.ndarray]:
    """Generate a few preprocessing variants to help capture end-of-line characters.

    - Otsu threshold (existing)
    - Adaptive threshold (Gaussian)
    - Slight dilation to connect faint strokes
    - Add white border to avoid edge truncation
    """
    variants: List[np.ndarray] = []
    h, w = img.shape[:2]
    base = img
    if max(h, w) < 900:
        base = cv2.resize(base, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 75, 75)

    # Attempt a light deskew
    try:
        # Stronger skew correction: use edges via Canny then minAreaRect
        edges = cv2.Canny(gray, 60, 180)
        pts = cv2.findNonZero(edges)
        if pts is not None and len(pts) > 200:
            rect = cv2.minAreaRect(pts)
            angle = rect[-1]
            if angle < -45:
                angle = 90 + angle
            if abs(angle) > 0.3 and abs(angle) < 20:
                (h2, w2) = gray.shape[:2]
                M = cv2.getRotationMatrix2D((w2 // 2, h2 // 2), angle, 1.0)
                base = cv2.warpAffine(base, M, (w2, h2), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                gray = cv2.warpAffine(gray, M, (w2, h2), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        pass

    # Otsu
    _, th_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(_add_border(th_otsu, 18))

    # Adaptive Gaussian
    th_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 9)
    variants.append(_add_border(th_adapt, 18))

    # Light dilation on Otsu to help thin end strokes
    kernel = np.ones((1, 2), np.uint8)
    th_dil = cv2.dilate(th_otsu, kernel, iterations=1)
    variants.append(_add_border(th_dil, 18))

    return variants


def _tesseract_extract_words(pil_img: Image.Image, psm: int = 6, conf_thresh: int = 50) -> List[str]:
    if pytesseract is None:
        return []
    cfg = f"--oem 3 --psm {psm}"
    try:
        data = pytesseract.image_to_data(pil_img, config=cfg, output_type=pytesseract.Output.DICT)
    except Exception:
        return []
    words: List[str] = []
    for t, c in zip(data.get('text', []), data.get('conf', [])):
        if not t or not t.strip():
            continue
        try:
            conf = int(float(c))
        except Exception:
            conf = -1
        if conf >= conf_thresh:
            words.append(t.strip())
    return words


def _tesseract_to_string(pil_img: Image.Image, psm: int = 6) -> str:
    if pytesseract is None:
        return ""
    cfg = f"--oem 3 --psm {psm} -l eng"
    try:
        txt = pytesseract.image_to_string(pil_img, config=cfg)
        return txt or ""
    except Exception:
        return ""


_easyocr_reader = None


def _easyocr_read(img: np.ndarray) -> List[str]:
    global _easyocr_reader
    try:
        import easyocr
    except Exception:
        return []
    if _easyocr_reader is None:
        try:
            _easyocr_reader = easyocr.Reader(['en'], gpu=False)
        except Exception:
            return []
    try:
        out = _easyocr_reader.readtext(img, detail=0)
        return [o.strip() for o in out if o and o.strip()]
    except Exception:
        return []


def _strip_timestamps(text: str) -> str:
    """Remove common timestamp patterns from OCR text."""
    if not text:
        return text
    # remove patterns like 12:22, 12:22:12, 12:22:12 PM, 12:22:12HM etc.
    text = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm|HM|hm)?\b", "", text)
    # remove standalone dates/times like 2025-09-12
    text = re.sub(r"\b\d{4}-\d{1,2}-\d{1,2}\b", "", text)
    # compact whitespace
    return re.sub(r"\s+", " ", text).strip()


def _dehyphenate_lines(text: str) -> str:
    """Join hyphenated line breaks like 'exam-\nple' -> 'example'."""
    if not text:
        return text
    # common hyphen+newline patterns
    text = re.sub(r"-\s*\n\s*", "", text)
    # Normalize newlines and spaces
    return re.sub(r"\s+", " ", text).strip()


def _detect_droidcam_overlay(text: str) -> bool:
    """Detect if text likely comes from a DroidCam overlay/header.

    Heuristic: presence of 'droid' and 'cam' or variants plus a timestamp.
    """
    if not text:
        return False
    s = text.lower()
    # common misspellings/spacing
    if ("droid" in s or "droi" in s) and "cam" in s:
        return True
    # phrases like 'droidcam', 'droid cam', 'droid cam video', 'video feed'
    if re.search(r"droid\s*cam|droidcam|video\s*feed", s):
        return True
    # if a timestamp exists in the text it's likely an overlay
    if re.search(r"\d{1,2}:\d{2}(?::\d{2})?", s):
        # but ignore pure content that looks like time only if other words exist
        return True
    return False


def _postprocess_text_and_maybe_rerun(best: str, frame: np.ndarray) -> str:
    """Post-process OCR result and, if overlay detected, rerun OCR on cropped frame.

    Strategy:
    - Strip timestamps from the best result.
    - If result looks like a DroidCam overlay (header/timestamp), crop out the top
      ~15% of the frame and attempt OCR again on the cropped image. Use same
      tesseract/easyocr sequence (fast full-frame attempt).
    """
    cleaned = _strip_timestamps(best)
    if not _detect_droidcam_overlay(best):
        return cleaned

    # overlay detected: crop out top banner and retry on remainder
    h, w = frame.shape[:2]
    crop_y = int(h * 0.15)  # remove top 15% where overlays usually sit
    crop = frame[crop_y:h, 0:w]
    if crop is None or crop.size == 0:
        return cleaned

    # Try a quick full-frame OCR on the cropped image
    proc = _preprocess_crop(crop)
    pil = _to_pil(cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR))
    words = _tesseract_extract_words(pil, psm=6, conf_thresh=50)
    if words:
        result = ' '.join(words)
        return _strip_timestamps(re.sub(r"\s+", " ", result).strip())

    # easyocr fallback
    easy = _easyocr_read(crop)
    if easy:
        return _strip_timestamps(' '.join(easy))

    return cleaned


def read_text_from_frame(
    frame: np.ndarray,
    boxes: Optional[List[dict]] = None,
    debug_dir: Optional[str] = None,
    use_cnn: Optional[bool] = False,
    **_: object,
) -> str:
    """Extract readable text from a frame.

    Preferred order with Gemini available:
    - Try Gemini once on the best crop (a provided box if any, else center crop, else full frame)
    - If Gemini fails or is not configured, fall back to legacy Tesseract/EasyOCR pipeline

    Legacy order (fallback): provided boxes -> center crop -> full frame -> EasyOCR fallback.
    When debug_dir is set, crops are saved for inspection.
    """
    _ensure_tesseract_on_windows()

    # Note: `use_cnn` is accepted for backward-compatibility with older
    # call sites. This implementation uses a robust Tesseract pipeline with
    # EasyOCR fallback regardless of this flag, so we ignore it safely.

    def _save(img_np: np.ndarray, name: str) -> None:
        if not debug_dir:
            return
        try:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, f"{name}.png"), img_np)
        except Exception:
            pass

    candidates: List[str] = []

    # Determine a single "best" crop for Gemini to control cost/latency.
    def _gemini_target() -> np.ndarray:
        """Choose the frame region for Gemini OCR.

        Strategy change: Use FULL FRAME first for richer context. If boxes exist and one
        box covers >35% of area, use that (assuming it's a document). Otherwise full frame.
        """
        h, w = frame.shape[:2]
        area_total = h * w
        if boxes:
            try:
                # find largest box area ratio
                best = None
                best_ratio = 0.0
                for b in boxes:
                    xy = b.get('xyxy') if isinstance(b, dict) else b
                    x1, y1, x2, y2 = map(int, xy)
                    box_area = max(0, (x2 - x1)) * max(0, (y2 - y1))
                    ratio = box_area / float(area_total + 1)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best = (x1, y1, x2, y2)
                if best and best_ratio > 0.35:
                    x1, y1, x2, y2 = best
                    pad = 12
                    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
                    x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
                    crop = frame[y1:y2, x1:x2]
                    _save(crop, 'gemini_doc_box')
                    return crop
            except Exception:
                pass
        _save(frame, 'gemini_full')
        return frame

    # Try Gemini first when configured
    if GEMINI_API_KEY:
        try:
            target = _gemini_target()
            txt = _call_gemini_transcribe(target)
            if txt:
                txt = re.sub(r"\r", "\n", txt)
                # Preserve original line structure for Gemini (don't dehyphenate yet)
                cleaned = re.sub(r"\n{2,}", "\n", txt).strip()
                try:
                    print(f"[OCR] Gemini used. Length={len(cleaned)}")
                except Exception:
                    pass
                if OCR_GEMINI_DIRECT:
                    return cleaned
                # Legacy overlay handling (if direct disabled)
                cleaned2 = _dehyphenate_lines(re.sub(r"\s+", " ", cleaned).strip())
                return _postprocess_text_and_maybe_rerun(cleaned2, frame)
        except Exception:
            try:
                print("[OCR] Gemini failed, using legacy OCR fallback.")
            except Exception:
                pass
            pass  # fallback to legacy path

    # Legacy OCR path below
    # 1) try provided boxes
    if boxes:
        for i, b in enumerate(boxes):
            try:
                xy = b.get('xyxy') if isinstance(b, dict) else b
                x1, y1, x2, y2 = map(int, xy)
                # Expand box slightly to avoid cutting off end characters
                pad_x = max(5, int((x2 - x1) * 0.03))
                pad_y = max(5, int((y2 - y1) * 0.08))
                x1e = max(0, x1 - pad_x)
                y1e = max(0, y1 - pad_y)
                x2e = min(frame.shape[1], x2 + pad_x)
                y2e = min(frame.shape[0], y2 + pad_y)
                crop = frame[y1e:y2e, x1e:x2e]
            except Exception:
                continue
            if crop is None or crop.size == 0:
                continue
            _save(crop, f'box_{i}')
            # Try multiple preprocess variants and PSMs
            for var in _preprocess_variants(crop):
                pil = _to_pil(cv2.cvtColor(var, cv2.COLOR_GRAY2BGR))
                txt1 = _tesseract_to_string(pil, psm=6)
                txt2 = _tesseract_to_string(pil, psm=3)
                for t in (txt1, txt2):
                    t = t.strip()
                    if t:
                        candidates.append(t)

    # 2) center crop
    h, w = frame.shape[:2]
    cw, ch = int(w * 0.5), int(h * 0.25)
    x1 = max(0, (w - cw) // 2)
    y1 = max(0, (h - ch) // 2)
    center = frame[y1:y1 + ch, x1:x1 + cw]
    if center is not None and center.size:
        _save(center, 'center')
        for var in _preprocess_variants(center):
            pil = _to_pil(cv2.cvtColor(var, cv2.COLOR_GRAY2BGR))
            txt1 = _tesseract_to_string(pil, psm=6)
            txt2 = _tesseract_to_string(pil, psm=3)
            for t in (txt1, txt2):
                t = t.strip()
                if t:
                    candidates.append(t)

    # 3) full frame
    _save(frame, 'full')
    # Add horizontal padding for full frame before variants to capture edge chars
    pad_w = int(frame.shape[1] * 0.04)
    padded_full = cv2.copyMakeBorder(frame, 0, 0, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(255,255,255))
    for var in _preprocess_variants(padded_full):
        pil = _to_pil(cv2.cvtColor(var, cv2.COLOR_GRAY2BGR))
        txt1 = _tesseract_to_string(pil, psm=6)
        txt2 = _tesseract_to_string(pil, psm=3)
        for t in (txt1, txt2):
            t = t.strip()
            if t:
                candidates.append(t)

    # 4) easyocr fallback
    if not candidates:
        easy = _easyocr_read(frame)
        if easy:
            candidates.append(' '.join(easy))

    if not candidates:
        return ''

    best = max(candidates, key=lambda s: len(s))
    best = _dehyphenate_lines(best)
    best = re.sub(r'\s+', ' ', best).strip()
    return _postprocess_text_and_maybe_rerun(best, frame)
