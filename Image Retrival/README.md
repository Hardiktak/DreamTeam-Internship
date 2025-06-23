# 🎯 Face Recognition Web App — Event Photo Matcher

A web-based face recognition system built during my internship at **DreamTeam Technologies** that helps users instantly retrieve their event photos using a single query image (selfie or webcam capture).

Unlike traditional face classification models, this system uses **deep face embeddings + FAISS** for high-speed similarity search, and it runs entirely offline — ensuring both performance and privacy.

---

## 🔍 Features

- Upload a selfie or click one via webcam
- Find all matching faces from a local image dataset
- Instant download of all results as a ZIP
- CLI-based real-time matching using webcam
- Fast similarity search using FAISS
- Deep embeddings from InsightFace
- Privacy-focused: No cloud, no APIs

---

## 🛠 Tech Stack

- Python
- Flask
- OpenCV
- InsightFace
- FAISS
- NumPy
- Scikit-learn

---

## 📁 Project Structure

├── app.py # Flask web app
├── encode_faces.py # Preprocessing script to encode dataset
├── face_matcher.py # Real-time CLI-based face matcher using webcam
├── embeddings.npy # Saved face embeddings
├── filenames.pkl # Corresponding image filenames
├── event_photos/ # Dataset of event images (subfolders per person)
├── static/
│ ├── uploads/ # Uploaded query images
│ ├── matched/ # Output images matched by the system
│ └── matched_faces.zip
└── templates/
└── index.html # Frontend page for image upload


---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/face-matcher-webapp.git
cd face-matcher-webapp
```

### 2. Create a Virtual Environment & Install Requirements
```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Prepare Your Dataset
- Place all event images inside the event_photos/ directory.
- Organize them in subfolders (e.g., per person or category).

### 4. Encode Face Embeddings
```bash
python encode_faces.py
```

### 5.  Run the Web App

```bash
python app.py
```

- Go to http://127.0.0.1:5000/ in your browser
- Upload a selfie (query image)
- Get all matching event photos with one click
- Download matched photos as a ZIP file

NOTE: Privacy Note
This project is fully offline:
- All images stay local (no cloud upload)
- Matching and inference is done on your machine
- Ideal for use in events, colleges, and private photo retrieval systems
