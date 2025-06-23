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




