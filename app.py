from flask import Flask, request, jsonify
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv
import os
import csv
from datetime import datetime

load_dotenv()

app = Flask(__name__)

def log_conversation(question, answer):
    with open("logs.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), question, answer])

with open("faq.txt", "r") as f:
    FAQ_CONTENT = f.read()

SYSTEM_PROMPT = f"""You are a helpful customer support assistant for 
Sharma's Coaching Center in Koramangala, Bangalore.
Answer ONLY using this FAQ information:

{FAQ_CONTENT}

If you don't know the answer, say: 
'I don't have that information. Please call us at 9876543210.'
Keep answers short and friendly."""

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

@app.route("/")
def home():
    return jsonify({"status": "Agent is running"})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    if not data or "message" not in data:
        return jsonify({"error": "Send a message field"}), 400
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=data["message"])
    ]
    result = llm.invoke(messages)
    log_conversation(data["message"], result.content)
    return jsonify({"reply": result.content})

@app.route("/whatsapp", methods=["POST"])
def whatsapp():
    incoming = request.values.get("Body", "").strip()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=incoming)
    ]
    result = llm.invoke(messages)
    log_conversation(incoming, result.content)
    resp = MessagingResponse()
    resp.message(result.content)
    return str(resp)

if __name__ == "__main__":
    app.run(port=5000, debug=True)