from flask import Flask, render_template, request, send_file, jsonify
import os
import numpy as np
import cv2
import pickle
import faiss
from sklearn.preprocessing import normalize
from werkzeug.utils import secure_filename
from zipfile import ZipFile
import shutil
import insightface

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
MATCHED_FOLDER = 'static/matched'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MATCHED_FOLDER, exist_ok=True)

# Load face model
model = insightface.app.FaceAnalysis(name='buffalo_l')
model.prepare(ctx_id=0)

# Load database
embeddings = np.load("embeddings.npy")
with open("filenames.pkl", "rb") as f:
    filenames = pickle.load(f)
embeddings = normalize(embeddings, axis=1)
index = faiss.IndexFlatIP(512)
index.add(embeddings)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Clean matched folder
    shutil.rmtree(MATCHED_FOLDER)
    os.makedirs(MATCHED_FOLDER, exist_ok=True)

    # Get uploaded images
    files = request.files.getlist("images")
    for f in files:
        f.save(os.path.join(UPLOAD_FOLDER, secure_filename(f.filename)))

    # Get query face
    query = request.files.get("query")
    if not query:
        return jsonify({"error": "No query image"}), 400

    img_bytes = np.frombuffer(query.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    faces = model.get(img)
    if not faces:
        return jsonify({"error": "No face detected"}), 400

    # Match face
    query_embedding = normalize(faces[0].embedding.reshape(1, -1), axis=1)
    D, I = index.search(query_embedding, len(filenames))
    threshold = 0.3
    matched_files = []

    for score, idx in zip(D[0], I[0]):
        if score < threshold:
            continue
        src_path = os.path.join("event_photos", filenames[idx])
        if os.path.isfile(src_path):
            dst_path = os.path.join(MATCHED_FOLDER, os.path.basename(filenames[idx]))
            shutil.copy(src_path, dst_path)
            matched_files.append(os.path.basename(dst_path))

    return jsonify({"matches": matched_files})

@app.route('/download')
def download():
    zip_path = "static/matched_faces.zip"
    with ZipFile(zip_path, 'w') as zipf:
        for filename in os.listdir(MATCHED_FOLDER):
            zipf.write(os.path.join(MATCHED_FOLDER, filename), arcname=filename)
    return send_file(zip_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
