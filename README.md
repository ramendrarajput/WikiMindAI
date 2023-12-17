# WikiMindAI
WikiMindAI - Wikipedia-based Mindful Artificial Intelligence

### Overview
WikiMindAI is a powerful conversational AI system that utilizes the vast knowledge of Wikipedia to provide insightful and informative answers to user queries. It is designed to facilitate exploration of topics, encourage curiosity, and foster learning through interactive conversations.

With WikiMindAI, users can explore a wide range of subjects, ask questions about specific topics, and receive detailed and informative answers directly from Wikipedia. The AI leverages the advanced capabilities of the DistilBERT model, making it efficient and accurate in handling a diverse set of questions.

### Features
- Topic Exploration: Users can enter a search topic, and WikiMindAI will provide them with a concise summary of the topic based on the first search result from Wikipedia. This allows users to get a quick overview and understanding of the subject matter.

- Question-Answering: Users can ask specific questions related to the selected topic, and WikiMindAI will provide detailed and informative answers based on the content from Wikipedia. It can intelligently analyze the context (paragraph) to deliver relevant and accurate responses.

- Efficient Caching: WikiMindAI implements caching to enhance performance and improve user experience. Frequently accessed data, such as the Question-Answering pipeline, is cached to reduce computation time and optimize response speed.

- Search Suggestion: WikiMindAI implements a search suggestion feature that provides autocomplete suggestions as the user types their search query. This can help users find relevant topics quickly and easily.

### Limitations
- Limited Translations: WikiMindAI only provides answers in English. This is because the load_qa_pipeline function uses the distilbert-base-uncased-distilled-squad model, which is specifically trained for English question-answering.
### How to Use WikiMindAI
1. Search Topic: Enter a search topic in the provided text input box labeled "Search Topic." Press the Enter key to submit the query.

2. View Topic Summary: After entering a topic, WikiMindAI will display a concise summary of the topic based on the first Wikipedia search result. This provides users with a quick understanding of the subject matter.

3. Ask Questions: Use the "Question" text input box to ask specific questions related to the selected topic. Enter your question and press the Enter key to get a detailed answer.

4. Observe Results: The AI will process your question and display the relevant answer based on the information from Wikipedia. The answer will be shown directly below the question input.

> Note that this Model is still under development and might not perform effectively. I'm trying to bring it up to speed as soon as possible
### Creator
Gideon Ogunbanjo
> [Credits](https://www.youtube.com/@eniolaa)