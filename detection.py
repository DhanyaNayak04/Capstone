"""Object and obstacle detection thread.

Adjustments:
- Refined obstacle detection to reduce false positives (was triggering on most objects).
    Uses central region of the depth map and a ratio of 'near' pixels relative to median depth
    instead of any single close pixel vs global mean.
"""
import time, cv2, torch, numpy as np
from models import yolo_model, midas, transform, device
from state import (
    latest_frame, frame_lock, latest_objects, latest_boxes, context_lock, obstacle_detected,
    obstacle_warning_state, detection_active
)
from tts import speak


def detection_thread():
    global latest_objects, obstacle_detected
    frame_count = 0
    print("[DETECTION] Thread started.")
    while True:
        if not detection_active[0]:
            time.sleep(0.1)
            continue
        with frame_lock:
            frame = latest_frame[0]
        if frame is None:
            time.sleep(0.05)
            continue
        frame_count += 1
        if frame_count % 2 == 0:
            results = yolo_model(frame)
            names = yolo_model.names
            detected = set()
            boxes_list = []
            h_frame, w_frame = frame.shape[:2]
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = names[cls_id]
                    detected.add(label)
                    # box.xyxy format: tensor of shape (4,)
                    xyxy = box.xyxy[0].cpu().numpy() if hasattr(box, 'xyxy') else None
                    if xyxy is not None:
                        x1, y1, x2, y2 = xyxy
                        area = max(0.0, (x2 - x1) * (y2 - y1))
                        boxes_list.append({
                            'label': label,
                            'xyxy': (float(x1), float(y1), float(x2), float(y2)),
                            'area': float(area),
                            'rel_area': float(area) / (w_frame * h_frame)
                        })
            with context_lock:
                # IMPORTANT: mutate the shared set/list in-place so other modules
                # holding a reference (imported from state) see updates.
                latest_objects.clear()
                latest_objects.update(detected)
                latest_boxes.clear()
                latest_boxes.extend(boxes_list)
            print(f"[DETECTION] Objects: {detected} boxes={len(boxes_list)}")
        if frame_count % 3 == 0:
            # Depth estimation
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_batch = transform(img_rgb).to(device)
            with torch.no_grad():
                prediction = midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=frame.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            depth_map = prediction.cpu().numpy()

            # Focus on central region (reduces edges / background influence)
            h, w = depth_map.shape
            ch, cw = int(h * 0.5), int(w * 0.5)
            y0, x0 = (h - ch) // 2, (w - cw) // 2
            roi = depth_map[y0:y0 + ch, x0:x0 + cw]

            # Median-based relative threshold (MiDaS depth is relative, not metric)
            median_depth = np.median(roi)
            if median_depth <= 0:
                near_ratio = 0.0
            else:
                near_mask = roi < (median_depth * 0.7)  # 30% closer than median
                near_ratio = near_mask.mean()

            # Decide obstacle if enough of central region is close
            obstacle_now = near_ratio > 0.08  # 8% of pixels close

            # Also consider large bounding boxes in central region as obstacle
            large_central_box = False
            try:
                cx_min, cy_min = x0, y0
                cx_max, cy_max = x0 + cw, y0 + ch
                for b in latest_boxes:
                    x1, y1, x2, y2 = b['xyxy']
                    # box centroid
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    # relative area threshold for "large" object
                    if b.get('rel_area', 0.0) >= 0.05 and (cx_min <= cx <= cx_max) and (cy_min <= cy <= cy_max):
                        large_central_box = True
                        break
            except Exception:
                large_central_box = False

            # combine signals: require either a sizeable central object OR a near_ratio
            obstacle_now = obstacle_now or large_central_box

            with context_lock:
                obstacle_detected[0] = bool(obstacle_now)

            if obstacle_now:
                if not obstacle_warning_state['active']:
                    obstacle_warning_state['active'] = True
                    obstacle_warning_state['count'] = 0
                if obstacle_warning_state['count'] < 2:
                    print(f"[DETECTION] ⚠️ Obstacle ahead (near_ratio={near_ratio:.2f})")
                    speak("Warning! Obstacle detected ahead.", priority=1)
                    obstacle_warning_state['count'] += 1
            else:
                if obstacle_warning_state['active']:
                    print("[DETECTION] Obstacle cleared.")
                obstacle_warning_state['active'] = False
                obstacle_warning_state['count'] = 0
        time.sleep(0.05)
