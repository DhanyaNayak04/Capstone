# facenet_recognition.py

import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from tts import speak
import os

# --- INITIALIZATION (No changes needed here) ---
# Initialize face detector and FaceNet
face_detector = MTCNN()
embedder = FaceNet()

EMBEDDINGS_FILE = 'known_faces_embeddings.npy'
NAMES_FILE = 'known_faces_names.npy'

def build_known_faces():
    known_faces = {}
    # Make sure these image files exist in the same directory
    image_paths = [('Disha', 'disha.jpg'), ("Dhanya", "dhanya.jpg"), ("Siri", "siri.jpg")]
    for name, img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"[Warning] Image path not found, skipping: {img_path}")
            continue
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # MTCNN expects RGB
        faces = face_detector.detect_faces(img)
        if faces:
            x, y, w, h = faces[0]['box']
            face_img = img[y:y+h, x:x+w]
            embedding = embedder.embeddings([face_img])[0]
            known_faces[name] = embedding
        else:
            print(f"No face found in {img_path}")
    return known_faces

if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(NAMES_FILE):
    embeddings = np.load(EMBEDDINGS_FILE)
    names = np.load(NAMES_FILE, allow_pickle=True)
    known_faces = dict(zip(names, embeddings))
else:
    print("Building known faces embeddings...")
    known_faces = build_known_faces()
    if known_faces:
        np.save(EMBEDDINGS_FILE, np.array(list(known_faces.values())))
        np.save(NAMES_FILE, np.array(list(known_faces.keys())))

# --- MODIFIED recognize_face FUNCTION ---
def recognize_face(frame):
    """
    Detects and recognizes faces in a frame, then speaks the results.
    This function is now designed to be called by other threads.
    """
    # MTCNN expects RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_detector.detect_faces(frame_rgb)
    
    recognized_names = []
    unknown_count = 0

    if not faces:
        speak("I don't see anyone in front of you.")
        return

    for face in faces:
        x, y, w, h = face['box']
        # Ensure coordinates are positive
        x, y = max(0, x), max(0, y)
        face_img = frame_rgb[y:y+h, x:x+w]
        
        if face_img.size == 0:
            continue

        embedding = embedder.embeddings([face_img])[0]
        
        name = 'Unknown'
        min_dist = float('inf')
        
        # Compare with known faces
        for k, v in known_faces.items():
            dist = np.linalg.norm(embedding - v)
            if dist < min_dist and dist < 0.9:  # Threshold can be tuned
                min_dist = dist
                name = k
        
        if name != 'Unknown':
            recognized_names.append(name)
        else:
            unknown_count += 1

    # --- Announce the results ---
    if recognized_names and unknown_count > 0:
        speak(f"I see {', '.join(recognized_names)} and {unknown_count} other person.")
    elif recognized_names:
        speak(f"I see {', '.join(recognized_names)}.")
    elif unknown_count > 1:
        speak(f"There are {unknown_count} unknown people in front of you.")
    elif unknown_count == 1:
        speak("There is an unknown person in front of you.")

def recognize_face_and_get_text(frame) -> str:
    """
    Detects and recognizes faces in a frame, then RETURNS the results as text.
    This function is for the API server and does NOT import/call speak().
    """
    # MTCNN expects RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_detector.detect_faces(frame_rgb)
    
    recognized_names = []
    unknown_count = 0

    if not faces:
        return "I don't see anyone in front of you."

    for face in faces:
        x, y, w, h = face['box']
        # Ensure coordinates are positive
        x, y = max(0, x), max(0, y)
        face_img = frame_rgb[y:y+h, x:x+w]
        
        if face_img.size == 0:
            continue

        embedding = embedder.embeddings([face_img])[0]
        
        name = 'Unknown'
        min_dist = float('inf')
        
        # Compare with known faces
        for k, v in known_faces.items():
            dist = np.linalg.norm(embedding - v)
            if dist < min_dist and dist < 0.9:  # Threshold can be tuned
                min_dist = dist
                name = k
        
        if name != 'Unknown':
            recognized_names.append(name)
        else:
            unknown_count += 1

    # --- Announce the results ---
    if recognized_names and unknown_count > 0:
        return f"I see {', '.join(recognized_names)} and {unknown_count} other person."
    elif recognized_names:
        return f"I see {', '.join(recognized_names)}."
    elif unknown_count > 1:
        return f"There are {unknown_count} unknown people in front of you."
    elif unknown_count == 1:
        return "There is an unknown person in front of you."
    
    return "I was unable to recognize any faces clearly."


# --- CRITICAL FIX: Place test loop inside if __name__ == '__main__' ---
if __name__ == '__main__':
    """
    This block will only run when you execute `python facenet_recognition.py` directly.
    It will NOT run when this file is imported by `realtime_ai_glasses.py`.
    """
    print("Running face recognition in standalone test mode...")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # In test mode, we want to see the visual output
        # So we create a modified version of the function that returns the frame
        
        # --- Copy of the logic but with drawing for visual feedback ---
        frame_rgb_test = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces_test = face_detector.detect_faces(frame_rgb_test)
        for face in faces_test:
            x, y, w, h = face['box']
            face_img_test = frame_rgb_test[y:y+h, x:x+w]
            embedding_test = embedder.embeddings([face_img_test])[0]
            name_test = 'Unknown'
            min_dist_test = float('inf')
            for k, v in known_faces.items():
                dist = np.linalg.norm(embedding_test - v)
                if dist < min_dist_test and dist < 0.9:
                    min_dist_test = dist
                    name_test = k
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name_test, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        cv2.imshow('Face Recognition Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()