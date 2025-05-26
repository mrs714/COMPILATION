import os
import zipfile
import json
import xml.etree.ElementTree as ET

print("Setting torch up...")
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import torch

# Configuration
ZIP_PATH = 'GenLayer/RAG test/data/usc.zip'
EXTRACT_TO = 'usc'
GENERAL_PATH = 'GenLayer/RAG test/data/'
JSON_PATH = 'GenLayer/RAG test/data/data.json'
INDEX_PATH = 'GenLayer/RAG test/data/index_faiss.bin'
CHUNKS_PATH = 'GenLayer/RAG test/data/index_chunks.json'
METADATA_PATH = 'GenLayer/RAG test/data/index_metadata.json'
CHUNK_SIZE = 2048
BATCH_SIZE = 10
TOP_K = 2

# Check if GPU is available and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")



# Initialize SentenceTransformer model with GPU support if available
#embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

def parse_xml_file(file_path):
    try:
        # Parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Define the namespace map
        namespaces = {
            'default': 'http://xml.house.gov/schemas/uslm/1.0',
            'dc': 'http://purl.org/dc/elements/1.1/',
            'dcterms': 'http://purl.org/dc/terms/'
        }

        # Function to recursively extract text from an element and its children
        def extract_text(element):
            text = element.text or ""
            for child in element:
                text += extract_text(child)
            return text

        # Function to find elements and return their concatenated text
        def find_elements_text(parent, tag, namespace='default'):
            elements = parent.findall(f'.//{{{namespaces[namespace]}}}{tag}')
            return [extract_text(element) for element in elements if element is not None]

        # Example usage
        section_texts = find_elements_text(root, 'section')
        title_text = find_elements_text(root, 'title', namespace='dc')

        return section_texts, title_text
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return [], []

def extract_zip(zip_path, extract_to):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")

def process_xml_files(directory, max_files=None):
    data = {}
    for file in os.listdir(directory):
        if file.endswith('.xml'):
            if max_files is not None and len(data) >= max_files:
                break
            section_texts, title_text = parse_xml_file(os.path.join(directory, file))
            data[file] = {
                'section_texts': section_texts,
                'title_text': title_text
            }
            print(f"Processed {file}: {len(section_texts)} sections, title: {title_text}")
    return data

def save_to_json(data, json_path):
    try:
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error saving to {json_path}: {e}")

def run_extraction_process(zip_path = ZIP_PATH, extract_to = EXTRACT_TO, json_path = JSON_PATH, max_files = None):
    # Check if GenLayer/RAG test/data/ directory exists, if not create it
    if not os.path.exists('GenLayer/RAG test/data/'):
        os.makedirs('GenLayer/RAG test/data/')

    # Extract the xml files from the zip archive
    extract_zip(zip_path, extract_to)

    # Parse the xml files to obtain the data
    data = process_xml_files(extract_to, max_files)

    print(f"Total files processed: {len(data)}")

    # Dump to the specified JSON path
    save_to_json(data, json_path)

def create_embedding_db(data_path = JSON_PATH, chunk_size=CHUNK_SIZE, batch_size=BATCH_SIZE):
    """
    Create an embedding database from text data.

    Args:
        data_path (str): Path to the JSON file containing text data.
        chunk_size (int): Size of each text chunk.
        batch_size (int): Number of chunks to process in each batch.
    """
    # Load and split US code into chunks from a JSON file
    with open(data_path) as f:
        data = json.load(f)

    chunks = []
    metadata = []

    print("Processing files and creating chunks...")
    for file_key, content in data.items():
        # Combine all section texts into a single string
        combined_text = " ".join(content["section_texts"])

        # Split into chunks
        file_chunks = [combined_text[i:i + chunk_size] for i in range(0, len(combined_text), chunk_size)]
        chunks.extend(file_chunks)

        # Store metadata for each chunk
        metadata.extend([(file_key, i) for i in range(len(file_chunks))])

    print(f"Total chunks created: {len(chunks)}")

    # Create embeddings for each chunk with batch processing
    print("Creating embeddings for chunks in batches...")
    text_embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_embeddings = embedding_model.encode(batch_chunks, convert_to_tensor=True)
        text_embeddings.extend(batch_embeddings.cpu().numpy())
        print(f"Embeddings created for chunks {i+1}-{min(i + batch_size, len(chunks))}/{len(chunks)}")

    text_embeddings = np.array(text_embeddings)

    # Store embeddings in FAISS
    d = text_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(text_embeddings)

    # Store everything in JSON files
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f)

    with open(CHUNKS_PATH, "w") as f:
        json.dump(chunks, f)

    faiss.write_index(index, INDEX_PATH)

def vectorize_query_path(query, embedding_model, index_path=INDEX_PATH, chunks_path=CHUNKS_PATH, metadata_path=METADATA_PATH, top_k=TOP_K):
    """
    Vectorize a query and retrieve similar chunks from the FAISS index.

    Args:
        query (str): The query text to vectorize.
        index_path (str): Path to the FAISS index file.
        chunks_path (str): Path to the chunks JSON file.
        metadata_path (str): Path to the metadata JSON file.
        top_k (int): Number of similar chunks to retrieve.

    Returns:
        tuple: Retrieved chunks and their metadata.
    """
    # Load FAISS index and data
    index = faiss.read_index(index_path)
    with open(chunks_path, "r") as f:
        chunks = json.load(f)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Create embedding for the query
    query_embedding = embedding_model.encode([query], convert_to_tensor=True).cpu().numpy()

    # Retrieve similar chunks
    D, I = index.search(query_embedding, k=top_k)
    retrieved_chunks = [chunks[i] for i in I[0]]
    retrieved_metadata = [metadata[i] for i in I[0]]

    return retrieved_chunks, retrieved_metadata

def vectorize_query(query, embedding_model, index, chunks, metadata, top_k=5):
    """
    Vectorize a query and retrieve similar chunks from the FAISS index.

    Args:
        query (str): The query text to vectorize.
        embedding_model (SentenceTransformer): The embedding model to use.
        index (faiss.Index): The FAISS index to search.
        chunks (list): List of text chunks.
        metadata (list): List of metadata corresponding to the chunks.
        top_k (int): Number of similar chunks to retrieve.

    Returns:
        tuple: Retrieved chunks and their metadata.
    """

    # Create embedding for the query
    query_embedding = embedding_model.encode([query], convert_to_tensor=True).cpu().numpy()

    # Retrieve similar chunks
    D, I = index.search(query_embedding, k=top_k)
    retrieved_chunks = [chunks[i] for i in I[0]]
    retrieved_metadata = [metadata[i] for i in I[0]]

    return retrieved_chunks, retrieved_metadata

def create_chunk_db(index_path, chunks_path, metadata_path):
    """
    This function will do the same as the create_embedding_db function, but it will save the index, chunks, and metadata in pieces to the specified paths.
    This is useful for large datasets that cannot be processed in one go.

    Args:
        index_path (str): Path to the folder that stores all the index files.
        chunks_path (str): Path to the folder that stores all the chunks.
        metadata_path (str): Path to the metadata JSON file.
    """
    
    # Load and split US code into chunks from a JSON file
    with open(JSON_PATH) as f:
        data = json.load(f)

    chunks = []
    metadata = []

    print("Processing files and creating chunks...")
    for file_key, content in data.items():
        # Combine all section texts into a single string
        combined_text = " ".join(content["section_texts"])

        # Split into chunks
        file_chunks = [combined_text[i:i + CHUNK_SIZE] for i in range(0, len(combined_text), CHUNK_SIZE)]
        chunks.extend(file_chunks)

        # Store metadata for each chunk
        metadata.extend([(file_key, i) for i in range(len(file_chunks))])

    print(f"Total chunks created: {len(chunks)}")

    # Create embeddings for each chunk with batch processing
    print("Creating embeddings for chunks in batches...")
    text_embeddings = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[i:i + BATCH_SIZE]
        batch_embeddings = embedding_model.encode(batch_chunks, convert_to_tensor=True)
        text_embeddings.extend(batch_embeddings.cpu().numpy())
        print(f"Embeddings created for chunks {i+1}-{min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    text_embeddings = np.array(text_embeddings)

    # Store embeddings in FAISS
    d = text_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(text_embeddings)

    # Store everything by pieces in JSON files
    chunk_count = len(chunks)
    chunk_size = 1000  # Number of chunks to save in each file
    num_files = (chunk_count + chunk_size - 1) // chunk_size  # Calculate the number of files needed

    for i in range(num_files):
        start_index = i * chunk_size
        end_index = min(start_index + chunk_size, chunk_count)
        chunk_file_path = os.path.join(chunks_path, f"chunks_part_{i}.json")
        metadata_file_path = os.path.join(metadata_path, f"metadata_part_{i}.json")
        
        # Save chunks and metadata for this part
        with open(chunk_file_path, "w") as f:
            json.dump(chunks[start_index:end_index], f)

        with open(metadata_file_path, "w") as f:
            json.dump(metadata[start_index:end_index], f)

    # Store the FAISS index distributedly
    index_chunks = 20
    index_chunk_size = (d + index_chunks - 1) // index_chunks  # Calculate the size of each index chunk
    for i in range(index_chunks):
        start_index = i * index_chunk_size
        end_index = min(start_index + index_chunk_size, d)
        index_file_path = os.path.join(index_path, f"index_part_{i}.bin")
        
        index_part = faiss.IndexFlatL2(end_index - start_index)
        index_part.add(text_embeddings[:, start_index:end_index])
        faiss.write_index(index_part, index_file_path)

    print("All chunks, metadata, and index parts have been saved.")


def load_chunk_db(index_path, chunks_path, metadata_path):
    """
    Load the chunk database from the specified paths.

    Args:
        index_path (str): Path to the folder that stores all the index files.
        chunks_path (str): Path to the folder that stores all the chunks.
        metadata_path (str): Path to the metadata JSON file.

    Returns:
        tuple: Loaded index, chunks, and metadata.
    """
    # Load FAISS index
        # Load FAISS index parts and concatenate them
    index_files = sorted([f for f in os.listdir(index_path) if f.endswith('.bin')])
    index_parts = []

    for file_name in index_files:
        index_part = faiss.read_index(os.path.join(index_path, file_name))
        index_parts.append(index_part)

    # Concatenate index parts horizontally
    dims = sum(part.d for part in index_parts)
    full_index = faiss.IndexFlatL2(dims)

    # Merge the vectors from all parts
    vectors = []
    for i in range(index_parts[0].ntotal):
        full_vector = np.hstack([part.reconstruct(i) for part in index_parts])
        vectors.append(full_vector)
    full_index.add(np.array(vectors))
    
    # Load chunks and metadata from JSON files
    chunks = []
    metadata = []
    for file_name in os.listdir(chunks_path):
        if file_name.endswith('.json'):
            with open(os.path.join(chunks_path, file_name), "r") as f:
                chunks.extend(json.load(f))

    for file_name in os.listdir(metadata_path):
        if file_name.endswith('.json'):
            with open(os.path.join(metadata_path, file_name), "r") as f:
                metadata.extend(json.load(f))

    return full_index, chunks, metadata

if __name__ == "__main__":
    # Create the chunk database (this will save the index, chunks, and metadata in pieces)
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    create_chunk_db('GenLayer/RAG test/RAG_server/store/',
                 'GenLayer/RAG test/RAG_server/store/',
                 'GenLayer/RAG test/RAG_server/store/')

# the index faiss is used to store the embeddings of the chunks, and it is used to retrieve the most similar chunks to a given query.
# The chunks are stored in a json file, and the metadata is stored in another json file.
# The metadata contains the file name and the index of the chunk in the original file.