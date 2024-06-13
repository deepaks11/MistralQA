import torch
# from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import time, os


c_w_d = os.getcwd()
dataset = os.path.join(c_w_d, "dataset/5th.pdf")
model = os.path.join(c_w_d, "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the file path to the PDF file
loader = PyMuPDFLoader(file_path=dataset)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
text_chunks = text_splitter.split_documents(data)

# Initialize Large Language Model for answer generation
llm_answer_gen = LlamaCpp(
    streaming=True,
    model_path=model,
    temperature=0.95,
    max_tokens=50,
    n_threads=8,
    n_gpu_layers=35,
    top_p=1,
    verbose=False,
    n_ctx=4096
)

# Create vector database for answer generation
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": device})

# Initialize vector store for answer generation
vector_store = Chroma.from_documents(text_chunks, embeddings)

# Initialize retrieval chain for answer generation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
answer_gen_chain = ConversationalRetrievalChain.from_llm(llm=llm_answer_gen, retriever=vector_store.as_retriever(), memory=memory)
# answer_gen_chain = RetrievalQA.from_chain_type(llm=llm_answer_gen, chain_type="stuff",
#                                                retriever=vector_store.as_retriever())

while True:

    user_input = input("Enter a question: ")
    s = time.time()
    if user_input.lower() == 'exit':
        break

    # Run question generation chain
    question = user_input
    answers = answer_gen_chain.run({"question": question})
    # answers = answer_gen_chain.run(question)

    # Print results outside the loop or limit the number of questions to process before printing
    print("Answer: ", answers)
    print("execute time :", time.time() - s)
