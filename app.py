import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed data
df = pd.read_excel("complete_preprocessing.xlsx")

# Ensure genre column is list
df['Genres Extracted'] = df['Genres Extracted'].apply(eval if isinstance(df['Genres Extracted'].iloc[0], str) else lambda x: x)

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Description'].fillna(""))
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['Book Name'].str.lower()).drop_duplicates()

# Clustering
kmeans = KMeans(n_clusters=10, random_state=42)
df['cluster'] = kmeans.fit_predict(tfidf_matrix)

# ------------------ Streamlit UI -------------------

st.title("📚 Book Recommendation System")
st.markdown("Get personalized book recommendations based on your preferences!")

# --- User Input
mode = st.radio("Choose Recommendation Mode", ["📘 By Favorite Book", "🎯 By Genre"])

if mode == "📘 By Favorite Book":
    book_input = st.text_input("Enter your favorite book title").strip().lower()

    if book_input:
        if book_input not in indices:
            st.warning("Book not found. Try another.")
        else:
            idx = indices[book_input]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:6]
            book_indices = [i[0] for i in sim_scores]
            recs = df.iloc[book_indices][['Book Name', 'Author', 'Rating', 'Genres Extracted']]

            st.subheader("🔮 Recommended Books:")
            st.dataframe(recs)

elif mode == "🎯 By Genre":
    all_genres = sorted(set(genre for sublist in df['Genres Extracted'] for genre in sublist))
    selected_genres = st.multiselect("Choose your favorite genres", all_genres)

    if selected_genres:
        mask = df['Genres Extracted'].apply(lambda x: any(g in x for g in selected_genres))
        recs = df[mask].sort_values(by='Rating', ascending=False).head(5)
        st.subheader("🔮 Recommended Books:")
        st.dataframe(recs[['Book Name', 'Author', 'Rating', 'Genres Extracted']])

# --- EDA / Visualization
st.markdown("---")
st.subheader("📊 Data Visualizations")

viz_option = st.selectbox("Choose a plot", ["Genre Distribution", "Rating Distribution", "Top Rated Books"])

if viz_option == "Genre Distribution":
    all_genres_flat = [genre for sublist in df['Genres Extracted'] for genre in sublist]
    genre_counts = pd.Series(all_genres_flat).value_counts().head(15)

    st.bar_chart(genre_counts)

elif viz_option == "Rating Distribution":
    fig, ax = plt.subplots()
    sns.histplot(df['Rating'], bins=20, kde=True, ax=ax)
    ax.set_title("Rating Distribution")
    st.pyplot(fig)

elif viz_option == "Top Rated Books":
    top_books = df.sort_values(by="Rating", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=top_books, x="Rating", y="Book Name", ax=ax)
    ax.set_title("Top 10 Rated Books")
    st.pyplot(fig)
