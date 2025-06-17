from wordcloud import WordCloud
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
from summarizer import summarize_article, analyze_sentiment, extract_entities, style_entity_tags
from topic_modeler import group_articles_by_topic
from news_fetcher import fetch_top_headlines
from datetime import datetime, timedelta
import pandas as pd
import io
import base64
import plotly.express as px
from collections import Counter

# --- Initialize Session State ---
if "bookmarked" not in st.session_state:
    st.session_state.bookmarked = []
if "last_fetch_time" not in st.session_state:
    st.session_state.last_fetch_time = None

# --- Streamlit App Config ---
st.set_page_config(page_title="AI News Summarizer", layout="wide")
st.title("📰 AI-Powered News Summarizer & Topic Grouper")
st.markdown("Fetches live news, groups them by topic, and summarizes each article using AI.")

# --- Sidebar Controls ---
st.sidebar.header("Configuration")
country = st.sidebar.selectbox("Select Country", options=["us", "gb", "de", "fr", "eg"], index=0)
category = st.sidebar.selectbox("News Category", options=["technology", "general", "sports", "business", "health"], index=0)
num_clusters = st.sidebar.slider("Number of Topic Clusters", min_value=2, max_value=10, value=3)

# --- Filters ---
st.sidebar.markdown("### Filters")
search_term = st.sidebar.text_input("Search keyword in title or summary", value="")
sentiment_filter = st.sidebar.selectbox("Filter by Sentiment", ["All", "Positive", "Negative"])

# --- Summarization Settings ---
st.sidebar.markdown("### Summarization Settings")
model_choice = st.sidebar.selectbox("Choose summarization model", ["facebook/bart-large-cnn", "t5-small", "facebook/mbart-large-cc25"])
language = st.sidebar.selectbox("Summarize in Language", ["English", "Arabic", "French", "German", "Spanish"], index=0)
max_len = st.sidebar.slider("Max Summary Length", 50, 250, 130)
min_len = st.sidebar.slider("Min Summary Length", 10, 100, 30)
show_sentiment = st.sidebar.checkbox("Show Sentiment", value=True)
show_entities = st.sidebar.checkbox("Show Entities", value=True)

# --- Auto-refresh Logic ---
refresh_interval = st.sidebar.selectbox("Auto-refresh interval (minutes)", [0, 5, 10, 15, 30, 60], index=0)
now = datetime.now()
time_to_refresh = (
    refresh_interval > 0 and
    (st.session_state.last_fetch_time is None or now - st.session_state.last_fetch_time > timedelta(minutes=refresh_interval))
)

# --- Fetch Trigger ---
if st.sidebar.button("🔄 Fetch News") or time_to_refresh:
    with st.spinner("Fetching and processing articles..."):
        st.session_state.last_fetch_time = now  # Update fetch time

        articles = fetch_top_headlines(
            api_key=st.secrets["news_api_key"],
            country_code=country,
            category=category
        )

        export_data = []
        entity_counter = Counter()

        if articles:
            grouped_articles, cluster_df = group_articles_by_topic(articles, num_clusters=num_clusters)
            grouped_articles.sort(key=lambda x: x[0])

            filtered_articles = []
            for cluster_id, article in grouped_articles:
                content = article.get('content') or article.get('description') or ""
                summary = summarize_article(
                    text=content,
                    model_name=model_choice,
                    max_length=max_len,
                    min_length=min_len
                )
                sentiment, score = analyze_sentiment(content) if content else ("", 0)

                if search_term.lower() not in article['title'].lower() and search_term.lower() not in summary.lower():
                    continue
                if sentiment_filter == "Positive" and sentiment != "POSITIVE":
                    continue
                if sentiment_filter == "Negative" and sentiment != "NEGATIVE":
                    continue

                filtered_articles.append((cluster_id, article, summary, sentiment, score, content))

            st.markdown("### 📊 Summarized News")
            progress = st.progress(0)
            total = len(filtered_articles)

            current_cluster = -1
            for idx, (cluster_id, article, summary, sentiment, score, content) in enumerate(filtered_articles):
                progress.progress((idx + 1) / total)

                if cluster_id != current_cluster:
                    st.markdown(f"## 🧠 Topic Cluster {cluster_id + 1}")
                    current_cluster = cluster_id

                with st.expander(f"🔸 {article['title']} ({article['source']['name']})"):
                    st.write(f"[Read full article ↗️]({article['url']})")

                    if content:
                        entities = extract_entities(content)
                        entity_counter.update([ent[0] for ent in entities])

                        if show_sentiment:
                            sentiment_emoji = {
                                "POSITIVE": "🟢",
                                "NEGATIVE": "🔴",
                                "NEUTRAL": "⚪"
                            }.get(sentiment.upper(), "⚪")
                            st.markdown(f"**Sentiment:** {sentiment_emoji} {sentiment} ({score:.2f})")

                        cols = st.columns([0.85, 0.15])
                        with cols[0]:
                            st.success(f"📝 Summary:\n\n{summary}")
                        with cols[1]:
                            if st.button("⭐ Bookmark", key=f"bookmark-{article['url']}"):
                                st.session_state.bookmarked.append({
                                    "Title": article['title'],
                                    "Source": article['source']['name'],
                                    "URL": article['url'],
                                    "Summary": summary,
                                    "Sentiment": sentiment,
                                    "Score": score,
                                    "Entities": ", ".join([f"{text} ({label})" for text, label in entities]) if entities else ""
                                })
                                st.success("Bookmarked!")

                        if show_entities:
                            if entities:
                                st.markdown("**Entities:**", unsafe_allow_html=True)
                                st.markdown(style_entity_tags(entities), unsafe_allow_html=True)
                            else:
                                st.markdown("**Entities:** No named entities found.")

                        export_data.append({
                            "Cluster": cluster_id + 1,
                            "Title": article['title'],
                            "Source": article['source']['name'],
                            "URL": article['url'],
                            "Sentiment": sentiment,
                            "Sentiment_Score": round(score, 2),
                            "Summary": summary,
                            "Entities": ", ".join([f"{text} ({label})" for text, label in entities]) if entities else ""
                        })
                    else:
                        st.warning("No content available for summarization.")

            # --- Download Section ---
            if export_data:
                st.markdown("### 💾 Download Your Results")
                df = pd.DataFrame(export_data)

                # CSV
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                b64_csv = base64.b64encode(csv_buffer.getvalue().encode()).decode()
                csv_link = f'<a href="data:file/csv;base64,{b64_csv}" download="news_summaries.csv">📥 Download CSV</a>'
                st.markdown(csv_link, unsafe_allow_html=True)

                # JSON
                json_str = df.to_json(orient='records', indent=2)
                b64_json = base64.b64encode(json_str.encode()).decode()
                json_link = f'<a href="data:application/json;base64,{b64_json}" download="news_summaries.json">📥 Download JSON</a>'
                st.markdown(json_link, unsafe_allow_html=True)

            # --- Topic Map ---
            if not cluster_df.empty:
                st.markdown("## 🧭 Topic Map")
                fig = px.scatter(
                    cluster_df,
                    x="x", y="y",
                    color=cluster_df["cluster"].astype(str),
                    hover_data=["title", "url"],
                    labels={"color": "Cluster"},
                    title="Interactive Topic Visualization"
                )
                st.plotly_chart(fig, use_container_width=True)

            # --- Trending Entities ---
            if entity_counter:
                top_entities = entity_counter.most_common(10)
                entity_df = pd.DataFrame(top_entities, columns=["Entity", "Frequency"])

                st.markdown("## 🔥 Trending Topics")
                fig2 = px.bar(
                    entity_df,
                    x="Entity",
                    y="Frequency",
                    color="Frequency",
                    title="Top 10 Mentioned Entities in News",
                    text_auto=True
                )
                st.plotly_chart(fig2, use_container_width=True)

                # --- Word Cloud ---
                st.markdown("## ☁️ Named Entity Word Cloud")
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color="white",
                    colormap="tab10"
                ).generate_from_frequencies(entity_counter)

                fig_wc, ax = plt.subplots(figsize=(10, 4))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig_wc)
        else:
            st.error("❌ Failed to fetch articles.")

# --- Bookmarked ---
if st.session_state.bookmarked:
    st.markdown("## ⭐ Bookmarked Articles")
    for b in st.session_state.bookmarked:
        with st.expander(f"🔖 {b['Title']} ({b['Source']})"):
            st.write(f"[Read Full Article ↗️]({b['URL']})")
            st.success(f"📝 {b['Summary']}")
            st.markdown(f"**Sentiment:** {b['Sentiment']} ({b['Score']:.2f})")
            st.markdown(f"**Entities:** {b['Entities']}", unsafe_allow_html=True)

    st.markdown("### 📥 Download Bookmarked Articles")
    bookmark_df = pd.DataFrame(st.session_state.bookmarked)

    csv_buf = io.StringIO()
    bookmark_df.to_csv(csv_buf, index=False)
    b64_bookmarks = base64.b64encode(csv_buf.getvalue().encode()).decode()
    st.markdown(f'<a href="data:file/csv;base64,{b64_bookmarks}" download="bookmarked_articles.csv">📥 Download CSV</a>', unsafe_allow_html=True)
