# MistralQA: An Intelligent Document Question Answering System

**MistralQA** is an advanced Question Answering (QA) system that extracts information from documents and provides precise answers to user queries. This project leverages the Mistral-7B language model in a Retrieval-Augmented Generation (RAG) framework, integrating various components from the LangChain library to deliver accurate responses from large PDF documents.

## Features

- **Document Processing**: Loads PDF files and splits them into manageable chunks using `PyPDFLoader` and `RecursiveCharacterTextSplitter`.
- **Advanced Language Model**: Utilizes the Mistral-7B model, fine-tuned for instruction-based tasks, for generating nuanced and contextually accurate answers.
- **Efficient Embeddings and Retrieval**: Implements `HuggingFaceEmbeddings` and stores vector representations in a `Chroma` vector database for fast and scalable retrieval.
- **Conversational Memory**: Maintains context across multiple interactions using `ConversationBufferMemory`, supporting coherent and continuous dialogue.
- **Performance Optimization**: Adapts to available hardware, running efficiently on both CPU and GPU.

## How It Works

1. **Load and Split Document**: The system loads a PDF and splits it into chunks using LangChain components.
2. **Create Embeddings**: Converts text chunks into dense vector representations with `HuggingFaceEmbeddings`.
3. **Retrieve Relevant Chunks**: Uses a semantic search to find relevant text chunks based on user queries.
4. **Generate Answers**: Feeds retrieved chunks into the Mistral-7B model to generate accurate responses.
5. **Maintain Context**: Uses conversational memory to keep track of dialogue context, enhancing multi-turn interactions.

## Quick Start

### Prerequisites

- Python 3.9 or higher
- PyTorch with CUDA support (optional for GPU acceleration)
- Install required Python libraries with `pip install -r requirements.txt`

# Hardware Requirements

  - CPU: Minimum 32GB RAM and Intel i7 processor
  - GPU: Minimum RTX 3060 with 8GB RAM

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/deepaks11/MistralQA
   cd RAG_llamacpp_mistral
   ```

# Hardware Requirements

  - CPU: Minimum 32 GB RAM and Intel i7 processor
  - GPU: Minimum RTX 3060 with 8 GB RAM


### GPU and CPU Requirements

2.  **If you're using GPU, ensure you have the following dependencies installed**:
```bash
   langchain
   langchain-community
   transformers
   sentence-transformers
   conda install -c conda-forge faiss-gpu
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   set LLAMA_CUDA=on
   pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python==0.2.76/whl/cu124
   PyMuPDFLoader
   chromadb
```
 2.  **If you're using CPU, ensure you have the following dependencies installed**:
```bash
   langchain
   langchain-community
   transformers
   sentence-transformers
   faiss-cpu
   torch torchvision torchaudio
   llama-cpp-python
   PyMuPDFLoader
   chromadb
```


3. **Download the Mistral Model**:
   - Ensure the Mistral-7B model is downloaded and update the `model` path in the code to its location.

### Running the Project

```bash
python run_cpu.py or python run_gpu.py
```

- **Interactive Mode**: Enter your questions, and the system will provide answers based on the loaded document.
- **Exit**: Type `exit` to end the session.

## Usage Example

```bash
Enter a question: What are the main findings in the report?
Answer: The report highlights several key findings including...
```

## Future Enhancements

- **Multi-document Support**: Extend to query across multiple documents.
- **Enhanced Memory Management**: Improve conversational flow handling.
- **Model Integration**: Support additional language models.

## Contribution

Contributions are welcome! Fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **LangChain**: For robust document processing tools.
- **Hugging Face**: For pre-trained models and the transformers library.
- **Mistral AI**: For the Mistral-7B model.

---

You can now copy this entire block at once and paste it directly into your README file on GitHub.
