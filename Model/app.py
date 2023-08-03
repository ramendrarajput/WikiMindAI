import wikipediaapi
import streamlit as st
from transformers import pipeline
from tokenizers import Tokenizer
from languages import languages
from gtts import gTTS
from pydub import AudioSegment
from io import BytesIO
import speech_recognition as sr

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
    headers = {
        'User-Agent': 'WikiMindAI/1.0 (https://gideonogunbanjo.netlify.app)'
    }
    wiki_wiki = wikipediaapi.Wikipedia(language, headers=headers)
    try:
        page = wiki_wiki.page(query)
        summary = page.summary
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

def text_to_speech(text, language_code):
    """
    Converts text to speech in the specified language using gTTS.

    Args:
        text (str): The text to be converted to speech.
        language_code (str): The language code for the speech synthesis.

    Returns:
        AudioSegment: The audio segment containing the speech.
    """
    tts = gTTS(text, lang=language_code)
    mp3_data = BytesIO()
    tts.write_to_fp(mp3_data)
    mp3_data.seek(0)
    audio = AudioSegment.from_file(mp3_data, format="mp3")
    return audio

def recognize_speech(language_code):
    """
    Captures audio from the microphone and converts it into text using SpeechRecognition.

    Args:
        language_code (str): The language code for the speech recognition.

    Returns:
        str: The recognized text.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)

    try:
        st.write("Recognizing...")
        text = recognizer.recognize_google(audio, language=language_code)
        return text
    except sr.UnknownValueError:
        st.write("Could not understand audio.")
    except sr.RequestError as e:
        st.write(f"Error with the service; {e}")
    return ""

# Main application engine
if __name__ == '__main__':
    # Display name and title
    st.title("WikiMindAI - Wikipedia-based Mindful Artificial Intelligence")
    st.write("Explore Topics, Ask Questions, and Receive Informative Answers!")

    # Language Selection
    language = st.selectbox("Select Language", list(languages.keys()))

    # Language Voice Support Checkbox
    use_voice = st.checkbox("Enable Language Voice Support")

    # Topic Input
    topic = st.text_input("Search Topic:", "")

    # Article Paragraph
    article_paragraph = st.empty()

    # Question Input
    question = st.text_input("Question:", "")

    # Asks Question with Voice
    if st.button("Ask Question with Voice"):
        if use_voice:
            recognized_text = recognize_speech(languages[language])
            st.write(f"Recognized Question: {recognized_text}")
            question = recognized_text

    if topic:
        # Map selected language to language code
        language_code = languages[language]

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

            # Language Voice Support
            if use_voice and summary:
                audio = text_to_speech(summary, language_code)
                st.audio(audio.raw_data, format="audio/mp3")

# Footer with link
link = 'Created by [Gideon Ogunbanjo](https://gideonogunbanjo.netlify.app)'
st.markdown(link, unsafe_allow_html=True)
