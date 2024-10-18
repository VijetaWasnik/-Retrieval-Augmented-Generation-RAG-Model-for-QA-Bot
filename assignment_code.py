# Step 1: Setup and imports

import fitz  # PyMuPDF for PDF handling
import requests
import numpy as np
import os
from google.colab import drive

# Step 2: Mount Google Drive
drive.mount('/content/drive')

# Step 3: Function to extract text from PDFs
def extract_text_from_pdfs(pdf_paths):
    text = []
    for pdf_path in pdf_paths:
        document = fitz.open(pdf_path)
        for page in document:
            text.append(page.get_text())
    return text

# Step 4: Define PDF file paths in Google Drive
books_folder = '/content/drive/My Drive/ML Books/'
pdf_paths = [os.path.join(books_folder, f) for f in os.listdir(books_folder) if f.endswith('.pdf')]

# Step 5: Extract text from all PDFs
book_texts = extract_text_from_pdfs(pdf_paths)

# Step 6: Hugging Face API to generate embeddings
HUGGING_FACE_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": f"Bearer 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"}  # Replace with your Hugging Face API Token

def get_hugging_face_embeddings(text_chunks):
    # Ensure input is a list of strings
    if not isinstance(text_chunks, list) or not all(isinstance(item, str) for item in text_chunks):
        print("Error: Input must be a list of strings.")
        return []

    response = requests.post(HUGGING_FACE_API_URL, headers=headers, json={"inputs": text_chunks})

    if response.status_code == 200:
        embeddings = response.json()
        if isinstance(embeddings, list) and len(embeddings) > 0 and isinstance(embeddings[0], list):
            return embeddings
        else:
            print("Error: Unexpected response format from API")
            return []
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return []

# Step 7: Create embeddings for all extracted texts
book_embeddings = get_hugging_face_embeddings(book_texts)

# Step 8: Function to find the closest text based on embeddings
def find_closest_text(query, embeddings, texts):
    query_embedding = get_hugging_face_embeddings([query])

    if query_embedding and isinstance(query_embedding, list) and len(query_embedding) > 0:
        query_embedding = np.array(query_embedding[0])  # Convert to numpy array for distance computation
        distances = [np.linalg.norm(query_embedding - np.array(emb)) for emb in embeddings]
        closest_index = np.argmin(distances)
        return texts[closest_index]
    else:
        print("Error: Could not generate embedding for the query.")
        return None

# Step 9: Ask a question and retrieve the answer
user_query = "What is the primary algorithm used in machine learning?"
answer = find_closest_text(user_query, book_embeddings, book_texts)

if answer:
    print("Answer:", answer)
else:
    print("Could not find an answer to your question.")
