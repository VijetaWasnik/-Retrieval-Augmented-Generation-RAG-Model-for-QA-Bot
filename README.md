# RAG-based Document QA Bot

This project implements a QA bot using Retrieval-Augmented Generation (RAG) that allows users to upload PDFs and ask questions based on the content. The bot retrieves the most relevant text and provides an answer.

## Features

- **PDF Upload**: Users can upload PDF documents for processing.
- **Text Extraction**: The system extracts text from PDFs using PyMuPDF.
- **Embedding Generation**: Embeddings are generated using Hugging Face's sentence-transformers model.
- **Query Answering**: Based on the user's query, the system retrieves the most relevant document snippet.

## Setup and Usage

### Local Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/VijetaWasnik/-Retrieval-Augmented-Generation-RAG-Model-for-QA-Bot/qa-bot.git
    cd qa-bot
    ```

2. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

4. Open the app in your browser at `http://localhost:8501`.

### Colab Setup

1. Open the Colab notebook `RAG_QA_Colab.ipynb`.
2. Upload your PDFs and run the cells to process the text and ask questions.

### Docker Setup

1. Build the Docker image:
    ```bash
    docker build -t streamlit-qa-bot .
    ```

2. Run the Docker container:
    ```bash
    docker run -p 8501:8501 streamlit-qa-bot
    ```

3. Access the app in your browser at `http://localhost:8501`.

## Challenges Faced

- **Handling API Rate Limits**: We had to carefully manage API calls to the Hugging Face model to avoid exceeding rate limits.
- **Optimizing Text Extraction**: Large PDF files were split into smaller chunks to improve response times for embedding generation.
- **Embedding Generation**: Batch processing was implemented to minimize API response times.

## Future Improvements

- **Scaling the System**: Future iterations could include scaling the embedding and retrieval process by using a custom-trained model.
- **Better Chunking Strategy**: Implement a smarter text chunking strategy to handle large PDF documents.
- **Caching Mechanism**: Add a caching mechanism for faster retrieval of common queries.

## License

MIT License
