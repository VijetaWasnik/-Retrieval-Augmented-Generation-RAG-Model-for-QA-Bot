# app.py
import streamlit as st
import fitz  # PyMuPDF for PDF handling
import requests
import numpy as np

# Hugging Face API details
HUGGING_FACE_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": f"Bearer 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"}  # Replace with your Hugging Face API Token

# Function to extract text from uploaded PDFs
def extract_text_from_uploaded_pdfs(files):
    text_data = []
    for uploaded_file in files:
        document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in document:
            text += page.get_text()
        text_data.append(text)
    return text_data

# Function to generate embeddings using Hugging Face API
def get_hugging_face_embeddings(text_chunks):
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

# Function to find the closest text based on embeddings
def find_closest_text(query, embeddings, texts):
    query_embedding = get_hugging_face_embeddings([query])

    if query_embedding and isinstance(query_embedding, list) and len(query_embedding) > 0:
        query_embedding = np.array(query_embedding[0])
        distances = [np.linalg.norm(query_embedding - np.array(emb)) for emb in embeddings]
        closest_index = np.argmin(distances)
        return texts[closest_index]
    else:
        print("Error: Could not generate embedding for the query.")
        return None

# Streamlit App
st.title("PDF Question Answering App")

# Step 1: Upload PDF files
uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    extracted_texts = extract_text_from_uploaded_pdfs(uploaded_files)
    st.write("Text extracted from PDFs.")

    # Step 2: Generate embeddings for the extracted text
    book_embeddings = get_hugging_face_embeddings(extracted_texts)
    
    # Step 3: Input query and retrieve answer
    user_query = st.text_input("Ask a question")
    if user_query:
        answer = find_closest_text(user_query, book_embeddings, extracted_texts)
        if answer:
            st.write(f"Answer: {answer}")
        else:
            st.write("Could not find an answer.")
