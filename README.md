# 📚 Audible Insights: Intelligent Book Recommendation System

An **end-to-end intelligent book recommendation system** built with NLP, clustering, and machine learning to help users discover books they’ll love. Includes an interactive Streamlit web app deployed on AWS EC2.

---
## 📸 Demo Screenshot

![Preview](https://github.com/Manav2507/Book-Recoomendation-System-NLP-Machine-Learning/blob/main/6_1.png)


## 📌 Problem Statement

> In a world with thousands of audiobooks, how can we help users find the right book based on their preferences or previous listening history?

This project tackles that challenge by combining **NLP**, **clustering**, and **content-based recommendations** to suggest personalized books using Audible data.

---

## 🚀 Project Highlights

### ✅ Data Preprocessing
- Merged multiple datasets with common fields (book name, author, etc.)
- Cleaned missing values, removed duplicates, and normalized text
- Extracted and predicted **missing genres** using NLP techniques

### 🧠 NLP & Clustering
- TF-IDF vectorization of book descriptions
- KMeans clustering to group books by similarity
- Cosine similarity for content-based filtering

### 🤖 Recommendation Models
| Model Type | Description |
|------------|-------------|
| **Content-Based** | Recommends books with similar descriptions or genres |
| **Clustering-Based** | Suggests books within the same cluster |
| **Hybrid** | Combines clustering and content similarity for better personalization |

### 📊 Evaluation Metrics
- **Precision**: % of recommended genres that are relevant
- **Recall**: % of actual relevant genres that were recommended
- **RMSE**: Evaluates user rating prediction accuracy (for future use)

### 🌐 Streamlit Web App
- Search by book name or genre
- Get real-time recommendations
- View interactive EDA visualizations (genre trends, top-rated books, etc.)

### ☁️ Deployment
- App deployed on **AWS EC2** using Python virtual environment
- Optimized for scalable and real-time performance

---

## 🧩 Datasets Used

| File | Description |
|------|-------------|
| `Audible_Catlog.csv` | Book-level metadata: name, rating, author, price |
| `Audible_Catlog_Advanced_Features.csv` | Descriptions, listening time, genres |

---

## 📈 Visualizations

- 📊 Genre frequency charts
- ⭐ Rating & review distributions
- 🏆 Top-rated books per genre/cluster
- 🔍 Word clouds for popular genres (optional)

---

## 🛠️ Tech Stack

| Category     | Tools/Tech Used                                       |
|--------------|--------------------------------------------------------|
| Language     | Python 3.12                                            |
| Libraries    | Pandas, Scikit-learn, NLTK, Streamlit, Seaborn         |
| ML Models    | KMeans, TF-IDF, Cosine Similarity                      |
| Deployment   | AWS EC2, VirtualEnv                                    |
| Visualization| Streamlit, Matplotlib, Power BI (optional)             |

---
