import streamlit as st
from transformers import pipeline

# Load summarization model (cached for performance)
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()

# App UI
st.title("AI Text Summarizer")
st.write("Enter a long text below and get a concise summary!")

# Input text box
long_text = st.text_area("Enter text to summarize:", height=200)

# Sliders for summary length
max_length = st.slider("Max Summary Length", min_value=50, max_value=300, value=130)
min_length = st.slider("Min Summary Length", min_value=20, max_value=100, value=30)

# Button click
if st.button("Summarize"):
    if long_text.strip():
        with st.spinner("Generating summary..."):
            # Optional: limit input length to avoid errors
            text = long_text[:1000]

            summary = summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )

            st.subheader("Summary:")
            st.success(summary[0]['summary_text'])
    else:
        st.warning("Please enter some text to summarize.")
