# Flask RAG System

A simple Flask-based Retrieval Augmented Generation (RAG) system that allows users to send text prompts and receive relevant information retrieved from a knowledge base.

## Features

-RESTAPI for querying the RAG system

- Document ingestion and storage
- Text embedding generation
- Vector similarity search

## Quick Start

1. Clone the repository:

   \`\`\`bash

   git clone https://github.com/yourusername/flask-rag-system.git

   cd flask-rag-system

   \`\`\`
2. Run the setup script:

   \`\`\`bash

   chmod +x setup.sh

   ./setup.sh

   \`\`\`

   This will create a virtual environment, install dependencies, and start the Flask application.
3. The application will be running at `http://localhost:5000`

## API Usage

### Query the RAG System

\`\`\`

POST /query

\`\`\`

Request body:

\`\`\`json

{

  "prompt": "Your query text here"

}

\`\`\`

Response:

\`\`\`json

{

  "result": "Retrieved information based on your query"

}

\`\`\`
