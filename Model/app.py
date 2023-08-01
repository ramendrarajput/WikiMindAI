# Importing libraries
import torch
import wikipedia as wk
import transformers as tf
import streamlit as st

# Main application engine
if __name__ == '__main__':
    # Display name and titile
    st.title("WikiMindAI - Wikipedia-based Mindful Artificial Intelligence")
    st.write("Search topics, Ask questions, Get answers!")