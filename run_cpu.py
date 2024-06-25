import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import LlamaCpp
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyMuPDFLoader
import time, os

c_w_d = os.getcwd()
dataset = os.path.join(c_w_d, "dataset/5th.pdf")
model = os.path.join(c_w_d, "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the file path to the PDF file
s = time.time()

loader = PyMuPDFLoader(file_path=dataset)
data = loader.load()
print("execute time :", time.time() - s)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=64,
    separators=['\n\n', '\n', '(?=>\. )', ' ', '']
)

docs = text_splitter.split_documents(data)

# Initialize Large Language Model for answer generation
llm = LlamaCpp(
    model_path=model,
    temperature=0.95,
    max_tokens=50,
    n_threads=8,
    n_gpu_layers=35,
    top_p=1,
    # f16_kv=True,
    verbose=False,
    n_ctx=4096,
    stop=["Q:", "\n"],
    echo=True
)

# Create vector database for answer generation
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/average_word_embeddings_glove.6B.300d", model_kwargs={"device": device})

chain = load_qa_chain(llm, chain_type="stuff")

db = FAISS.from_documents(docs, embeddings)

while True:

    question = input("Enter a question: ")
    s = time.time()
    if question.lower() == 'exit':
        break

    docs = db.similarity_search(question)
    # Run question generation chain
    response = chain.run(
        input_documents=docs,
        question=question
    )

    # Print results outside the loop or limit the number of questions to process before printing
    print("Answer: ", response)
    print("execute time :", time.time() - s)
