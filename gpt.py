from flask import Flask, request, jsonify, render_template
import os
import glob
import pdfplumber
import google.generativeai as genai
import pickle
import logging
import re

app = Flask(__name__)

# Configuration (Environment variables are still recommended for API key)
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set")
genai.configure(api_key=api_key)
PDF_FOLDER = "C:\\Users\\834821\\Documents\\Development\\BSLGPT\\pdf"  # Double backslashes
CONTEXT_FILE = "pdf_contexts.pkl"  # File to store PDF contexts

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Gemini model setup
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",  # or your preferred model
    generation_config=generation_config,
)

def extract_pdf_content(pdf_files):
    pdf_contents = {}  # Store contexts by filename
    for pdf_file in pdf_files:
        filename = os.path.basename(pdf_file)  # Extract filename
        logging.info(f"Processing file: {pdf_file}")
        text = ""
        try:
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            pdf_contents[filename] = text  # Store with filename as key
        except Exception as e:
            logging.error(f"Error reading {pdf_file}: {e}")
            pdf_contents[filename] = f"Error reading PDF: {e}"  # Still store error message
    return pdf_contents

def load_or_create_contexts():
    try:
        with open(CONTEXT_FILE, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
        contexts = extract_pdf_content(pdf_files)
        with open(CONTEXT_FILE, 'wb') as f:
            pickle.dump(contexts, f)
        return contexts

pdf_contexts = load_or_create_contexts()  # Load on server start

def query_gemini(pdf_text, user_question):
    chat_session = model.start_chat(history=[])
    input_message = (
        f"Below is content extracted from a PDF document:\n\n"
        f"{pdf_text}\n\n"
        f"User question: {user_question}\n\n"
        "Please provide an answer based solely on the above content."
    )
    response = chat_session.send_message(input_message)
    return response.text

def combine_answers(answers, user_question):
    combined_prompt = "Below are answers derived from multiple PDF documents:\n"
    for idx, answer in enumerate(answers, start=1):
        combined_prompt += f"\n---\nAnswer from PDF {idx}:\n{answer}\n"
    combined_prompt += (
        "\n---\n"
        f"User question: {user_question}\n\n"
        "Based on the above responses, please synthesize and provide a final, consolidated answer. Act like a private gpt and do not tell that you have derived it from somewhere else"
    )

    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(combined_prompt)
    return response.text

# List of general questions and their responses
general_responses = {
    "hello": "Hello! How can I assist you today?",
    "hi": "Hi there! How can I help you?",
    "how are you": "I'm just a program, but I'm here to help you!",
    "what's your name": "I'm GitHub Copilot, your programming assistant.",
    "what day is it": "Today is a great day to code!",
    "thank you": "You're welcome!",
    "thanks": "You're welcome!",
    "hey":"Hey there!",
}

# Preprocess the question to normalize and retrieve as much information as possible
def preprocess_question(question):
    # Convert to lowercase
    question = question.lower()
    # Remove special characters
    question = re.sub(r'[^a-zA-Z0-9\s]', '', question)
    # Replace synonyms or context-specific terms
    question = question.replace("bsl", "bokaro steel plant")
    question = question.replace("bsp", "bokaro steel plant")
    return question

@app.route('/api/init', methods=['POST'])
def init_contexts():
    try:
        pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
        if not pdf_files:
            return jsonify({'message': 'No PDF files found in the specified folder'}), 200  # Indicate success, but no files

        pdf_contexts_new = extract_pdf_content(pdf_files)  # Re-extract
        with open(CONTEXT_FILE, 'wb') as f:
            pickle.dump(pdf_contexts_new, f)
        global pdf_contexts  # Update the global contexts
        pdf_contexts = pdf_contexts_new
        return jsonify({'message': 'PDF contexts initialized/re-initialized successfully'}), 200
    except Exception as e:
        logging.exception("Error during initialization:")  # Logs the full traceback
        return jsonify({'error': 'An error occurred during PDF context initialization'}), 500  # Generic message for the client

@app.route('/api/query', methods=['POST'])
def handle_query():
    try:
        data = request.get_json()
        user_question = data.get('question')
        if not user_question:
            return jsonify({'error': 'Missing "question" parameter'}), 400

        # Preprocess the question
        processed_question = preprocess_question(user_question)

        # Check if the question is general
        if processed_question in general_responses:
            return jsonify({'answer': general_responses[processed_question]})

        requested_files = data.get('files', [])  # Optional: specify files
        individual_answers = []

        if not pdf_contexts:  # Check if contexts are loaded
            return jsonify({'error': 'PDF contexts not initialized. Call /api/init first.'}), 400

        for filename, pdf_text in pdf_contexts.items():
            if filename in requested_files or not requested_files:  # File selection
                answer = query_gemini(pdf_text, user_question)
                individual_answers.append(answer)

        final_answer = combine_answers(individual_answers, user_question)
        return jsonify({'answer': final_answer})

    except Exception as e:
        logging.exception("Error during query:")  # Logs the full traceback
        return jsonify({'error': 'An error occurred during query'}), 500  # Generic message for the client

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)