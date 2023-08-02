import wikipedia as wk
import streamlit as st
from transformers import pipeline


# Page configuration
st.set_page_config(
    page_title="WikiMindAI"
)
    
def load_qa_pipeline():
    """
    Loads the Question-Answering pipeline using the DistilBERT model.

    Returns:
        Pipeline: The Question-Answering pipeline.
    """
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    return qa_pipeline


def load_wiki(query):
    """
    Searches Wikipedia for the given query and return a summary of the first search result.

    Args:
        query (str): The search query for Wikipedia.

    Returns:
        str: The summary of the first Wikipedia search result.
    """
    results = wk.search(query)
    summary = wk.summary(results[0], sentences=10)
    return summary


def answer_questions(pipeline, question, paragraph):
    """
    Uses the Question-Answering pipeline to answer a question based on the given context (paragraph).

    Args:
        pipeline (Pipeline): The Question-Answering pipeline.
        question (str): The question to be answered.
        paragraph (str): The context (paragraph) from which the question should be answered.

    Returns:
        dict: A dictionary containing the answer to the question and additional details.
    """
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
    st.write("Explore Topics, Ask Questions, and Receive Informative Answers!")

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
link='Created by [Gideon Ogunbanjo](https://gideonogunbanjo.netlify.app)'
st.markdown(link,unsafe_allow_html=True)