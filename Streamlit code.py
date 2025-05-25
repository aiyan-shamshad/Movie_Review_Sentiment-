import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer
from wordcloud import WordCloud
st.set_page_config(page_title="Movie Sentiment Analyzer", page_icon="🎬")

SENTIMENT_INFO = {
    0: {"label": "Negative", "emoji": "😠", "color": "#FF6B6B"},
    1: {"label": "Neutral", "emoji": "😐", "color": "#FFD166"}, 
    2: {"label": "Positive", "emoji": "😊", "color": "#06D6A0"}
}

@st.cache_resource
def load_model():
    return joblib.load('best_sentiment_model.pkl')

model = load_model()



st.title("🎬 Movie Review Sentiment Analysis")

review = st.text_area("Enter a movie review:", 
                     "The acting was great but the plot was disappointing...",
                     height=150)

if st.button("Analyze Sentiment"):
    with st.spinner("Analyzing..."):
        
        prediction = model.predict([review])[0]
        probabilities = model.predict_proba([review])[0]
        
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sentiment", f"{SENTIMENT_INFO[prediction]['emoji']} {SENTIMENT_INFO[prediction]['label']}")
        with col2:
            st.metric("Confidence", f"{probabilities[prediction]*100:.1f}%")
        with col3:
            st.metric("Certainty", "High" if max(probabilities) > 0.7 else "Medium")
        
        
        st.subheader("Sentiment Probability")
        fig, ax = plt.subplots()
        ax.bar(
            [SENTIMENT_INFO[i]['label'] for i in range(3)],
            probabilities,
            color=[SENTIMENT_INFO[i]['color'] for i in range(3)]
        )
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        
        st.subheader("Key Influencing Words")
        explainer = LimeTextExplainer(class_names=["Negative", "Neutral", "Positive"])
        exp = explainer.explain_instance(
            review, 
            model.predict_proba, 
            num_features=10
        )
        st.components.v1.html(exp.as_html(), height=400)