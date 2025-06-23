import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm
import insightface

# Initialize model
model = insightface.app.FaceAnalysis(name='buffalo_l')
model.prepare(ctx_id=0)  # CPU = -1 for CPU, 0 for GPU

# Paths
dataset_dir = "event_photos"
embedding_list = []
filename_list = []

# Process subfolders
for person in os.listdir(dataset_dir):
    person_path = os.path.join(dataset_dir, person)
    if not os.path.isdir(person_path):
        continue
    for filename in os.listdir(person_path):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        filepath = os.path.join(person_path, filename)
        img = cv2.imread(filepath)
        if img is None:
            print(f"⚠️ Failed to read {filepath}")
            continue
        faces = model.get(img)
        if not faces:
            print(f"❌ No face in: {filepath}")
            continue
        for face in faces:
            embedding_list.append(face.embedding)
            filename_list.append(os.path.join(person, filename))

# Save embeddings
np.save("embeddings.npy", np.array(embedding_list).astype("float32"))
with open("filenames.pkl", "wb") as f:
    pickle.dump(filename_list, f)

print(f"\n✅ Encoded {len(embedding_list)} faces from {len(set(filename_list))} images.")
