# ref-log.md

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

Generative AI prompts:

- Use the given documentation and give me all the learning details for all the technologies used.
- Explain RAG in great detail considering I am new to the concept.
- What are embeddings and what is OpenAIEmbeddings.
- How to make an llm model creative? -> A: using temperature.
- How can I clear vector store?
- When I click on clear vector store I want to drop all the uploaded files and the app should drop all the uploaded files and delete the collection, what tools can I use to do that? Give me all the documentation.
- How can I refresh the session state after popping a key.

Tools used:
Github Codespace, OpenAI API, Streamlit, Python Standard Libraries, LangChain, ChromaDB, dotenv, shutil.
