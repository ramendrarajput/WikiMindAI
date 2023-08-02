# Importing libraries
import torch
import wikipedia as wk
import transformers as tf
import streamlit as st
from transformers import pipeline, Pipeline


def load_qa_pipeline() -> Pipeline:
    qa_pipeline = pipeline("Question-Answering", model="distilbert-base-uncased-distilled-squad")
    return qa_pipeline


def load_wiki(query: str) -> str:
    results = wk.search(query)
    summary = wk.summary(results[0], sentences = 10)
    return summary
    

# Main application engine
if __name__ == '__main__':
    # Display name and titile
    st.title("WikiMindAI - Wikipedia-based Mindful Artificial Intelligence")
    st.write("Search topics, Ask questions, Get answers!")
    
    # Topic Input
    topic = st.text_input("Search Topic", "")
    
    # Article Paragraph
    article_paragraph = st.empty()
    # Question Input
    question = st.text_input("Question", "")
    
    if topic:
        # Loads Wikipedia summary of topic
        summary = load_wiki(topic)
        
        # Displays article summary in paragraph
        article_paragraph.markdown(summary)
        
        # -- Questions--
        if question is "":
            