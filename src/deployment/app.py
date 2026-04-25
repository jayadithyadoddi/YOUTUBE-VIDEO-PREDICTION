"""
Streamlit Web Application — YouTube Viral Predictor
Author: Mandaka Nadini
Run: streamlit run src/deployment/app.py
"""

import streamlit as st
import joblib
import pandas as pd
import os

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube Viral Predictor",
    page_icon="🎬",
    layout="centered",
)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "../../models/random_forest.pkl")
    return joblib.load(model_path)


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🎬 YouTube Viral Video Predictor")
st.markdown("Predict whether your YouTube video will go **viral** based on its features.")
st.markdown("---")

st.subheader("📝 Enter Video Details")

col1, col2 = st.columns(2)

with col1:
    title_length = st.slider("Title Length (characters)", 10, 150, 60)
    tag_count = st.slider("Number of Tags", 0, 50, 15)
    category_id = st.selectbox(
        "Category",
        options=[1, 10, 17, 22, 23, 24, 25, 28],
        format_func=lambda x: {
            1: "Film & Animation",
            10: "Music",
            17: "Sports",
            22: "People & Blogs",
            23: "Comedy",
            24: "Entertainment",
            25: "News & Politics",
            28: "Science & Technology",
        }.get(x, str(x)),
    )
    publish_hour = st.slider("Publish Hour (0-23)", 0, 23, 14)

with col2:
    likes = st.number_input("Expected Likes", min_value=0, value=5000, step=500)
    dislikes = st.number_input("Expected Dislikes", min_value=0, value=200, step=50)
    comment_count = st.number_input("Expected Comments", min_value=0, value=300, step=50)
    publish_dayofweek = st.selectbox(
        "Publish Day",
        options=list(range(7)),
        format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x],
    )

comments_disabled = st.checkbox("Comments Disabled?", value=False)
ratings_disabled = st.checkbox("Ratings Disabled?", value=False)
publish_month = st.slider("Publish Month", 1, 12, 6)

# ── Compute derived features ──────────────────────────────────────────────────
views_estimate = max(likes * 20, 1)
engagement_rate = min((likes + comment_count) / views_estimate, 1.0)
like_dislike_ratio = likes / (dislikes + 1)

# ── Predict ───────────────────────────────────────────────────────────────────
st.markdown("---")
if st.button("🚀 Predict Virality", use_container_width=True):
    try:
        model = load_model()

        features = {
            "likes": likes,
            "dislikes": dislikes,
            "comment_count": comment_count,
            "title_length": title_length,
            "tag_count": tag_count,
            "engagement_rate": engagement_rate,
            "like_dislike_ratio": like_dislike_ratio,
            "category_id": category_id,
            "comments_disabled": int(comments_disabled),
            "ratings_disabled": int(ratings_disabled),
            "publish_hour": publish_hour,
            "publish_dayofweek": publish_dayofweek,
            "publish_month": publish_month,
        }

        X = pd.DataFrame([features])
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]

        st.markdown("---")
        if prediction == 1:
            st.success(f"🔥 **VIRAL!** — Your video has a **{probability * 100:.1f}%** chance of going viral.")
        else:
            st.warning(f"📉 **Not Viral** — Viral probability: **{probability * 100:.1f}%**. Try optimizing your title, tags, or publish timing.")

        st.progress(float(probability))
        st.caption(f"Engagement Rate: {engagement_rate:.4f} | Like/Dislike Ratio: {like_dislike_ratio:.1f}")

    except FileNotFoundError:
        st.error("⚠️ Model file not found. Please run `src/models/train_model.py` first to train and save the model.")

st.markdown("---")
st.caption("Built by Mandaka Nadini · YouTube Video Performance Prediction Project · 2026")
