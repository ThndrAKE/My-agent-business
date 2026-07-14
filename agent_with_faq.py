from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

print("Loading FAQ document...")
loader = TextLoader("faq.txt")
documents = loader.load()

splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks = splitter.split_documents(documents)

print("Building knowledge base (takes 30 seconds first time)...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
knowledge_base = FAISS.from_documents(chunks, embeddings)

llm = ChatGroq(model="llama-3.3-70b-versatile")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=knowledge_base.as_retriever(),
    chain_type_kwargs={
        "prompt": None
    }
)

SYSTEM_PROMPT = """You are a helpful customer support assistant for 
Sharma's Coaching Center in Koramangala, Bangalore. 
Answer ONLY using the information provided to you.
If you don't know the answer, say: 
'I don't have that information. Please call us at 9876543210.'
Keep answers short and friendly. Always respond in the same 
language the customer uses."""

def ask(question):
    full_question = f"{SYSTEM_PROMPT}\n\nCustomer asks: {question}"
    result = qa_chain.invoke({"query": full_question})
    return result["result"]

print("\n✓ Agent ready! Ask anything (type 'quit' to exit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    response = ask(user_input)
    print(f"\nAgent: {response}\n")