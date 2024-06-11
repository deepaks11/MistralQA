# Intent Classification Using LangChain and LlamaCpp

This project uses LangChain integrated with the LlamaCpp model to perform intent classification through advanced text generation. Specifically, it employs the OpenChat-3.5-0106.Q4_K_M.gguf model. By leveraging various tools from LangChain, such as the RecursiveCharacterTextSplitter, FAISS vector stores, and HuggingFaceEmbeddings, the project constructs a comprehensive system for accurate intent identification. 

## Features

- Intent identification using text generation LLM model
- Utilizes LangChain's RecursiveCharacterTextSplitter
- Question-answering chain using LangChain
- Integration with LlamaCpp
- Vector stores using FAISS
- Embeddings with HuggingFaceEmbeddings
- CSV document loader

# System Requirements
  Ensure your system meets the following software requirements:
  - Operating System: Windows, macOS, or Linux
  - Python: Version 3.10 or higher
  - Conda: Anaconda 

# Hardware Requirements

  - CPU: Minimum 16 GB RAM and Intel i7 processor
  - GPU: Minimum RTX 3060 with 8 GB RAM

## Installation Instructions

To install and run this project locally, follow these steps:

1. **Clone the Repository**: First, clone the project repository to your local machine using Git:

```sh
   git clone https://github.com/deepaks11/intent_classification_llamacpp
   cd intent_classification_llamacpp

2. Set Up Conda Environment
   conda create --name (env name) python=3.10
   conda activate intent-identification

3. Install Dependencies
   pip install -r requirements.txt

4. Run the Project
   python run_cpu.py or run_gpu.py
```
## Dataset Format
   ```sh
   "</s>[INST] this intent about the server list and the intent associated with the server, can you give me a servers list,
    is there any server available right now, get the configured server details, give me the server details, How many servers have been configured,  
    Can you provide me with an updated list of all our servers [/INST] "intent": "Server List" </s>"
   ```
## CPU Requirements

To run this project on a CPU, you'll need the following Python packages:

```plaintext
langchain
langchain-community
transformers
sentence-transformers
faiss-cpu
torch
torchvision
torchaudio
llama-cpp-python
```
## GPU Requirements

For GPU support, you'll need the following:

1. Install the necessary Python packages:

```sh
  conda install -c conda-forge faiss-gpu
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```
2. If `llama-cpp-python` is not installed for GPU, refer to the [official installation guide](https://github.com/abetlen/llama-cpp-python).

## Links
- For more details on the integration with LlamaCpp, refer to the https://python.langchain.com/v0.2/docs/integrations/llms/llamacpp
- Download the model from Hugging Face and place it in the appropriate directory https://huggingface.co/TheBloke/openchat-3.5-0106-GGUF.
 
## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Author
**Deepak.s**
- [GitHub](https://github.com/deepaks11)
- [Email](mailto:deepaklsm11@gmail.com)

## Contributions

Contributions, issues, and feature requests are welcome!
                                                                       
