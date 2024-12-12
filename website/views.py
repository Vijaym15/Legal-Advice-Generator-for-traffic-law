from flask import Flask, Blueprint, render_template, request
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from .models import History
from . import db

views = Blueprint('views', __name__)

# Load models and components once during startup
checkpoint = "MBZUAI/LaMini-Flan-T5-783M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.float32)
embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")

# Load data from the text file
data_path = "E:\\Fall semester 2024-25\\SLP\\Project\\21BAI1421_LLM_LAG\\data\\Tamil_Nadu_Traffic_Police_Rules_and_Regulations.txt"  # Replace with your actual file path
loader = TextLoader(data_path)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Create Chroma vector database
persist_directory = "ipc_vector_data"
vector_db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)

pipe = pipeline(
    'text2text-generation',
    model=base_model,
    tokenizer=tokenizer,
    max_length=10000,
    do_sample=True,
    temperature=1,
    top_p=0.95
)
local_llm = HuggingFacePipeline(pipeline=pipe)
qa_chain = RetrievalQA.from_chain_type(
    llm=local_llm,
    chain_type='stuff',
    retriever=vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3}),  # Adjust k as needed
    return_source_documents=True,
)

def chat(chat_history, user_input):
    bot_response = qa_chain({"query": user_input})
    if not bot_response['source_documents']:
        response = "I'm sorry, I couldn't find information on that in the provided document."
    else:
        response = bot_response['result']
    chat_history.append((user_input, response))
    return chat_history

@views.route('/')
def home():
    return render_template("home.html")

@views.route('/input', methods=['POST', 'GET'])
def input_route():  # Renamed to avoid conflict with Python's input()
    if request.method == 'POST':
        chat_history = []
        user_input = request.form.get('input')
        chat_history = chat(chat_history, user_input)
        if chat_history:
            for prompt, response in chat_history:
                hist = History(user=prompt, bot=response)
                db.session.add(hist)
                db.session.commit()
        return render_template("output.html", chat_history=chat_history)
    return render_template("input.html")

@views.route('/history')
def history():
    items = History.query.all()
    return render_template("history.html", items=items)