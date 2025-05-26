from flask import Flask, request, jsonify
import os

app = Flask(__name__)

API_KEY = "set_here"

# Load the functions from embed_utils.py
from embed_utils import load_chunk_db, vectorize_query

print("Setting torch up...")
from sentence_transformers import SentenceTransformer
import torch

# Setup the system (load the data and start the model)

# Check if GPU is available and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize SentenceTransformer model with GPU support if available
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
print("Model loaded: all-MiniLM-L6-v2; Loading data...")

path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(path, "store")
index, chunk_db, metadata = load_chunk_db(path, path, path)

print("Data loaded. Starting server...")

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    if not data or 'prompt' not in data:
        return jsonify({"error": "No prompt provided"}), 400
    
    # Check that the key is valid
    if 'key' not in data or data['key'] != API_KEY:
        return jsonify({"error": "Invalid or non-existent API key"}), 401
    
    prompt = data['prompt']
    try:
        retrieved_chunks, retrieved_metadata = vectorize_query(prompt, embedding_model, index, chunk_db, metadata)
        return jsonify({"result": retrieved_chunks, "metadata": retrieved_metadata}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 80))
    app.run(host='0.0.0.0', port=port, debug=False)

    # wait and test
    # curl -X POST -H "Content-Type: application/json" -d '{"prompt": "test", "key": "set_here"}' http://localhost:5000/query