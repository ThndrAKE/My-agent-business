from flask import Flask, request, jsonify
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

app = Flask(__name__)

SYSTEM_PROMPT = """You are a helpful customer support assistant for 
Sharma's Coaching Center in Koramangala, Bangalore. 
Answer ONLY using the information provided to you.
If you don't know the answer, say: 
'I don't have that information. Please call us at 9876543210.'
Keep answers short and friendly."""

print("Loading FAQ and building knowledge base...")
loader = TextLoader("faq.txt")
documents = loader.load()
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks = splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
knowledge_base = FAISS.from_documents(chunks, embeddings)
llm = ChatGroq(model="llama-3.3-70b-versatile")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=knowledge_base.as_retriever()
)
print("Agent ready!")

@app.route("/")
def home():
    return jsonify({"status": "Agent is running"})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    if not data or "message" not in data:
        return jsonify({"error": "Send a message field"}), 400
    question = f"{SYSTEM_PROMPT}\n\nCustomer asks: {data['message']}"
    result = qa_chain.invoke({"query": question})
    return jsonify({"reply": result["result"]})

@app.route("/whatsapp", methods=["POST"])
def whatsapp():
    incoming = request.values.get("Body", "").strip()
    print(f"Received WhatsApp message: {incoming}")
    question = f"{SYSTEM_PROMPT}\n\nCustomer asks: {incoming}"
    result = qa_chain.invoke({"query": question})
    answer = result["result"]
    resp = MessagingResponse()
    resp.message(answer)
    return str(resp)

if __name__ == "__main__":
    app.run(port=5000, debug=True)