import os
import cv2
import tqdm
import numpy as np
import pickle
from deepface import DeepFace

# === CONFIGURATION ===
REFERENCE_DIR = "C:/Users/Jules/Desktop/reference faces"
OUTPUT_PKL = "representations_facenet_finalegit.pkl"
MODEL_NAME = "Facenet"
INPUT_SHAPE = (160, 160)

# === LOAD FACENET MODEL ===
print("[INFO] Loading ArcFace model...")
model = DeepFace.build_model(MODEL_NAME)

# === HELPER FUNCTION ===
def preprocess_face(img_path):
    try:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, INPUT_SHAPE)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        print(f"[ERROR] Failed processing {img_path}: {e}")
        return None

# === EMBEDDING EXTRACTION ===
representations = []

print("[INFO] Building embeddings...")

# Get the total number of images to process
total_images = sum([len(files) for root, dirs, files in os.walk(REFERENCE_DIR)])

# Initialize tqdm progress bar
with tqdm.tqdm(total=total_images, desc="Processing Embeddings", unit="image") as pbar:
    for root, dirs, files in os.walk(REFERENCE_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, file)
                
                # Use DeepFace's represent function for ArcFace
                try:
                    # Use DeepFace.represent() to extract embeddings
                    result = DeepFace.represent(
                        img_path=img_path,
                        model_name=MODEL_NAME,
                        enforce_detection=False,  # Skip face detection if faces are already cropped
                        detector_backend='skip'  # Use if you're already passing pre-cropped faces
                    )
                    embedding = result[0]["embedding"]

                    representations.append({
                        "identity": img_path,
                        "embedding": embedding
                    })
                    pbar.set_postfix({"Processed": img_path})
                except Exception as e:
                    print(f"[ERROR] Failed to process {img_path}: {e}")
                
                # Update progress bar
                pbar.update(1)

# === SAVE TO PKL ===
with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(representations, f)

print(f"[SUCCESS] Embeddings saved to: {OUTPUT_PKL}")
