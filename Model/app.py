import wikipedia as wk
import streamlit as st
from transformers import pipeline

def load_qa_pipeline():
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    return qa_pipeline

def load_wiki(query):
    results = wk.search(query)
    summary = wk.summary(results[0], sentences=10)
    return summary

def answer_questions(pipeline, question, paragraph):
    input_data = {
        "question": question,
        "context": paragraph
    }
    output = pipeline(input_data)
    return output

# Main application engine
if __name__ == '__main__':
    # Display name and title
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
        if question:
            # Loads the question answering pipeline
            qa_pipeline = load_qa_pipeline()

            # Answers query question using article summary
            result = answer_questions(qa_pipeline, question, summary)
            answer = result["answer"]

            # Displaying answer
            st.write(answer)
