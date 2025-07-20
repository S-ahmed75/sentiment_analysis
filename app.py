# Streamlit Sentiment Analysis App (Insights Only with Default File)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
import os
import altair as alt

# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(page_title="News Sentiment Analyzer", layout="wide")
st.title("📰 News Sentiment Analyzer (Insights View)")

# ----------------------------
# Load Data Automatically
# ----------------------------
def load_default_file():
    if os.path.exists("all-data.csv"):
        return pd.read_csv("all-data.csv", encoding='ISO-8859-1', header=None, names=["predicted_sentiment", "COMMENT"])
    return None

# ----------------------------
# File Upload or Default
# ----------------------------
st.sidebar.markdown("### 📁 Data Source")
uploaded_file = st.sidebar.file_uploader("Upload CSV (2 columns: sentiment, comment - no headers)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1', header=None, names=["predicted_sentiment", "COMMENT"])
elif load_default_file() is not None:
    df = load_default_file()
    st.success("Auto-loaded all-data.csv")
else:
    st.warning("Please upload a CSV file or make sure 'all-data.csv' exists in the directory.")
    st.stop()

# ----------------------------
# Data Preparation
# ----------------------------
df.drop_duplicates(inplace=True)

# ----------------------------
# Sentiment Distribution
# ----------------------------
st.subheader("📈 Sentiment Distribution")

sentiment_counts = df['predicted_sentiment'].value_counts().reset_index()
sentiment_counts.columns = ['Sentiment', 'Count']

bar_chart = alt.Chart(sentiment_counts).mark_bar().encode(
    x=alt.X('Sentiment', sort=['positive', 'neutral', 'negative']),
    y='Count',
    color=alt.Color('Sentiment', legend=None)
).properties(width=600, height=400)

text = bar_chart.mark_text(
    align='center',
    baseline='bottom',
    dy=-5
).encode(
    text='Count:Q'
)

st.altair_chart(bar_chart + text, use_container_width=True)

# ----------------------------
# Summary Insight
# ----------------------------
st.subheader("📊 Summary Insight")
total = len(df)
percentages = df['predicted_sentiment'].value_counts(normalize=True) * 100
dominant_sentiment = percentages.idxmax()
dominant_value = percentages.max()

st.markdown(f"""
- The dataset contains **{total}** comments.
- **{dominant_sentiment.capitalize()}** is the dominant sentiment with **{dominant_value:.2f}%** of the total.
- Here's the percentage breakdown:
""")

for sentiment, percent in percentages.items():
    st.markdown(f"- {sentiment.capitalize()}: {percent:.2f}%")

# ----------------------------
# Keyword Analysis (Unique by Dominant Sentiment)
# ----------------------------
st.subheader("🔍 Top Keywords per Sentiment (Unique by Dominant Sentiment)")
stop_words = set(STOPWORDS).union({"https", "co", "RT"})

vectorizer = CountVectorizer(stop_words='english', max_features=1000)

# Step 1: Word frequencies per sentiment
word_freq_by_sentiment = {}
for sentiment in ['positive', 'neutral', 'negative']:
    texts = df[df['predicted_sentiment'] == sentiment]['COMMENT'].dropna().astype(str)
    if not texts.empty:
        X = vectorizer.fit_transform(texts)
        sum_words = X.sum(axis=0)
        word_freq = {word: int(sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()}
        word_freq_by_sentiment[sentiment] = word_freq

# Step 2: Merge and assign to dominant sentiment
combined_freq = {}
for sentiment, freq_dict in word_freq_by_sentiment.items():
    for word, freq in freq_dict.items():
        if word not in combined_freq:
            combined_freq[word] = {}
        combined_freq[word][sentiment] = freq

# Step 3: Keep only dominant sentiment per word
final_top_words = {'positive': [], 'neutral': [], 'negative': []}
for word, sentiment_freqs in combined_freq.items():
    dominant_sentiment = max(sentiment_freqs, key=sentiment_freqs.get)
    final_top_words[dominant_sentiment].append((word, sentiment_freqs[dominant_sentiment]))

# Step 4: Sort & limit top N per sentiment
for sentiment in final_top_words:
    final_top_words[sentiment] = sorted(final_top_words[sentiment], key=lambda x: x[1], reverse=True)[:20]

# ----------------------------
# WordClouds (Based on Dominant Sentiment Only)
# ----------------------------
def generate_wordcloud_from_keywords(keyword_list, title, colormap):
    text = " ".join([word for word, _ in keyword_list])
    wordcloud = WordCloud(
        max_font_size=50,
        max_words=40,
        background_color="white",
        stopwords=stop_words,
        colormap=colormap
    ).generate(text)

    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.set_title(title, fontsize=16)
    ax.axis("off")
    st.pyplot(fig)

col1, col2 = st.columns(2)
with col1:
    generate_wordcloud_from_keywords(final_top_words['positive'], "Positive", "Greens")
with col2:
    generate_wordcloud_from_keywords(final_top_words['negative'], "Negative", "Reds")

# ----------------------------
# Display Keyword Tables
# ----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🟢 Positive Keywords")
    for word, freq in final_top_words['positive']:
        st.write(f"{word}: {freq}")

with col2:
    st.markdown("### ⚪ Neutral Keywords")
    for word, freq in final_top_words['neutral']:
        st.write(f"{word}: {freq}")

with col3:
    st.markdown("### 🔴 Negative Keywords")
    for word, freq in final_top_words['negative']:
        st.write(f"{word}: {freq}")

# ----------------------------
# Download Results
# ----------------------------
# Export only dominant sentiment top keywords
keyword_data = []
for sentiment, words in final_top_words.items():
    for word, freq in words:
        keyword_data.append({"Sentiment": sentiment, "Keyword": word, "Frequency": freq})

keywords_df = pd.DataFrame(keyword_data)
keywords_csv = keywords_df.to_csv(index=False).encode('utf-8')

st.subheader("⬇️ Download Top Keywords")
st.download_button(
    label="Download Keywords CSV",
    data=keywords_csv,
    file_name="top_keywords.csv",
    mime="text/csv"
)
