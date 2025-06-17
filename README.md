# 📰 AI News Summarizer & Topic Grouper

An AI-powered web app that automatically fetches the latest news, summarizes them, detects sentiment, extracts named entities, and visualizes topic clusters. Built with 🧠 Transformers, 📊 Plotly, and 🔥 Streamlit.

## 🔍 Features

- ✅ Live news fetching via NewsAPI
- ✅ Abstractive summarization using models like BART, T5, and MBART (multilingual)
- ✅ Sentiment analysis & named entity recognition (NER)
- ✅ Bookmarking system
- ✅ CSV/JSON export
- ✅ Interactive topic clustering & entity charts
- ✅ Auto-refresh news scheduling
- ✅ Word cloud of trending entities

## 🚀 Live Demo

> Coming soon: [Streamlit Cloud Deployment Link](https://share.streamlit.io/your-username/news-summarizer/main/app.py)

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/news-summarizer
cd news-summarizer
pip install -r requirements.txt
```

## 🔐 Setup API Key

Create a `.streamlit/secrets.toml` file:

```toml
news_api_key = "YOUR_NEWS_API_KEY_HERE"
```

## 💡 How It Works

- News articles are pulled from NewsAPI based on country & category.
- Articles are grouped using vector embeddings & clustering.
- Summaries are generated with Hugging Face Transformers (BART/T5/MBART).
- Sentiment and entities are analyzed and visualized.

## 📦 Tech Stack

- Streamlit
- Transformers (Hugging Face)
- spaCy
- scikit-learn
- Plotly
- WordCloud
- Pandas
- Matplotlib

## ✨ Screenshots

!Summary Preview ![image](https://github.com/user-attachments/assets/3b87b3c0-9fc7-4bea-b680-038e640b979a)
!Trending Topics ![image](https://github.com/user-attachments/assets/e21d958a-e467-420e-b028-ea8a857cd9cd)
!Topic Map ![image](https://github.com/user-attachments/assets/e39b785e-03bf-4350-b291-062c9fae0c8e)

## 🤝 License

MIT License © 2025
