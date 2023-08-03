import wikipediaapi
import streamlit as st
from transformers import pipeline
from tokenizers import Tokenizer

# Page configuration
st.set_page_config(
    page_title="WikiMindAI",
    layout="centered",
    initial_sidebar_state="collapsed"
)

@st.cache(hash_funcs={Tokenizer: lambda _: None}, allow_output_mutation=True)   
def load_qa_pipeline():
    """
    Loads the Question-Answering pipeline using the DistilBERT model.

    Returns:
        Pipeline: The Question-Answering pipeline.
    """
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    return qa_pipeline

def load_wiki(query, language="en"):
    """
    Searches Wikipedia for the given query in the specified language and returns a summary of the first search result.

    Args:
        query (str): The search query for Wikipedia.
        language (str): The language code for the Wikipedia search (default is "en" for English).

    Returns:
        str: The summary of the first Wikipedia search result.
    """
    wiki_wiki = wikipediaapi.Wikipedia(language)
    try:
        page = wiki_wiki.page(query)
        summary = page.summary[:500]  # Limit summary to 500 characters
        return summary
    # Disambiguation Error Exception
    except wikipediaapi.exceptions.DisambiguationError:
        return "Multiple articles found. Please provide a more specific topic."
    # Internet Connection Error 
    except wikipediaapi.exceptions.HTTPTimeoutError:
        return "No internet connection. Please check your internet settings."
    except Exception as e:
        return f"An Error Occurred: {e}"

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

    # Language Selection
    language = st.selectbox("Select Language", ["English", "Spanish"])

    # Topic Input
    topic = st.text_input("Search Topic:", "")

    # Article Paragraph
    article_paragraph = st.empty()

    # Question Input
    question = st.text_input("Question:", "")

    if topic:
        # Map selected language to language code
        language_code = "en"  # Default to English
        if language == "Spanish":
            language_code = "es"

        # Loads Wikipedia summary of topic in the selected language
        summary = load_wiki(topic, language=language_code)

        # Displays article summary in paragraph
        article_paragraph.markdown(summary)

        # -- Questions--
        if question:
            # Loads the question answering pipeline
            qa_pipeline = load_qa_pipeline()

            # Answers query question using article summary
            result = answer_questions(qa_pipeline, question, summary)
            answer = result["answer"]

            # Displaying answer in real-time
            st.write(answer)

# Footer with link
link = 'Created by [Gideon Ogunbanjo](https://gideonogunbanjo.netlify.app)'
st.markdown(link, unsafe_allow_html=True)
