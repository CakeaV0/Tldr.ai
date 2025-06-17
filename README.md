Tldr.ai is an AI-powered news summarizer and trend explorer for people who want the what, why, and who — without the scroll fatigue.

It fetches the latest global headlines, summarizes articles using advanced NLP models (BART, T5, mBART), groups related stories by topic clusters, analyzes sentiment, and highlights key named entities.

Features include:

Multilingual summarization

Real-time topic clustering

Sentiment tracking

Word cloud & trending entity visualizations

Bookmarking & one-click exports

Auto-refresh news scheduling

Whether you're a news junkie, analyst, or just want to stay updated in less time — Tldr.ai gets you the story behind the noise.


"# 📰 AI News Summarizer & Topic Grouper

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

![Summary Preview](screenshots/summary_ui.png)
![Entity Word Cloud](screenshots/wordcloud.png)
![Topic Map](screenshots/topic_map.png)

## 🤝 License

MIT License © 2025
"
