# Description

This is a RAG application where users can upload documents and interact with the content through a chatbot.

## Implementation Details & Design Choices

- The application uses `OPENAI` model to embed and process documents and user Questions.
- For model it uses `openai.gpt-5-chat` as it is the latest model from OPENAI and fastest chat model. For retaining larger contextual information the application uses `openai.text-embedding-3-large` embeddings. It also uses `temperature=0.2` to introduce some randomness in the behaviour of the model to generate unique responses.
- It uses `STREAMLIT` to provide frontend interface with various helper methods performing the backend operations.
- To read the documents the app uses `TextLoader` and `PyPDFLoader` provided by langchain.
- The App uses `CHROMADB` to store all the embeddings for processing the uploaded documents.
- It performs chunking of documents by using `RecursiveCharacterTextSplitter` with some overlap between the chunks for faster retrieval and contextual linking between chunks.
- For chunking by default it uses chunk size to be 1000. This was chosen based on information available online suggesting chunk size of 800-1200. Similarly Overlap of 150 was selected based on suggested range of 10%-20%.
- The application uses Persistent storage for efficient usage of memory.
- You can add .env file to make application changes to following parameters: `OPENAI_BASE_URL`, `PERSIST_DIR`, `COLLECTION`, `EMBED_MODEL`, `CHAT_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `FETCH_K`, `APP_TITLE`, `SYSTEM_PROMPT`.
- If .env is available application will load the variables, otherwise will use the default values.
- The app can clear the state of the chat to start a fresh chat without stopping the application.
- When the user asks a `question` about any details from the document, the application takes that query and pass the query along with some chat history and embeddings to generate a response.
- This response is then given back to the user.

## Steps to run Document RAG application

1. Make sure all the pip packages in the requirements.txt are installed using command `pip install -r requirements.txt`.

2. Then run: `pip install chromadb`.

3. To start the application run the command: `OPENAI_API_KEY=<your-api-key> streamlit run document_chat_app.py`.

4. Then open the link: `http://localhost:8501` in the browser.

5. Click on the "Browse files" button and upload `.txt`/`.pdf`/`.md` files, you can upload multiple documents.

6. Once you have uploaded the files you will see an "Index uploaded files" button, click on that to chunk and process uploaded files.

7. All the documents are indexed and these embeddings are stored in a local persistent Vector Store `.chroma`.

8. You can also click on "Clear vector & store uploaded files" button to clear the vector store collection and chat history, this will also remove all the uploaded documents.
