# Book-Recoomendation-System-NLP-Machine-Learning

## 📚 Audible Insights: Intelligent Book Recommendation System

An end-to-end book recommendation system using **NLP, Machine Learning, and Streamlit**, designed to help users discover books they’ll love based on genre preferences or previously liked titles. Deployed on **AWS EC2** for scalable access.

---

## 🚀 Project Overview

**Audible Insights** is an intelligent book recommender system that:
- Processes and cleans book datasets
- Extracts text-based features using NLP
- Applies clustering (K-Means) to group books by similarity
- Builds multiple recommendation models
- Offers an interactive web interface using Streamlit
- Deploys the application on AWS

---

## 🧠 Features

### ✅ Data Preprocessing
- Merges multiple datasets on common fields
- Cleans missing values, duplicates, and formatting issues
- Extracts and predicts missing genres from descriptions

### ✅ NLP & Clustering
- TF-IDF vectorization of book descriptions
- KMeans clustering for book grouping
- Cosine similarity for content-based filtering

### ✅ Recommendation Models
- **Content-Based Filtering**: recommends books based on similar descriptions/genres
- **Clustering-Based Filtering**: recommends books within the same cluster
- **Hybrid Model**: blends both techniques
- Evaluation using **Precision**, **Recall**, and **RMSE**

### ✅ Application UI (Streamlit)
- Search by **book title** or **genre**
- View personalized recommendations
- Explore interactive **EDA visualizations** (genre distributions, top books, etc.)

### ✅ Deployment
- App hosted on **AWS EC2**
- Built using a virtual environment with Python 3.12
- Optimized for real-time recommendations

---

## 🧩 Datasets Used

- `Audible_Catlog.csv`
- `Audible_Catlog_Advanced_Features.csv`

### Features include:
- Book Name, Author, Rating, Reviews, Price
- Description, Listening Time, Genres

---

## 💡 Use Cases

- 📖 Personalized reading recommendations
- 🏪 Enhancing digital bookstores & libraries
- 📊 Author & publisher analytics
- 💬 Discovering hidden gems by genre/theme

---

## 📈 Visualizations

- Genre frequency bar charts
- Ratings and reviews distributions
- Top-rated books per cluster or genre

---

## 🛠️ Tech Stack

- **Languages**: Python 3.12
- **Libraries**: Pandas, scikit-learn, NLTK, Streamlit, Matplotlib, Seaborn
- **Deployment**: AWS EC2 + Streamlit
- **Recommendation Models**: KMeans, Cosine Similarity (TF-IDF)

---

## 🧪 Evaluation Metrics

| Metric     | Description                                |
|------------|--------------------------------------------|
| Precision  | Relevant genres in top N recommendations   |
| Recall     | How many actual genres were predicted      |
| RMSE       | Used when integrating future user ratings  |
