from flask import Flask, request, jsonify
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from dotenv import load_dotenv
import os
import csv
from datetime import datetime

load_dotenv()

app = Flask(__name__)

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.getenv("TWILIO_FROM")
TWILIO_TO = os.getenv("TWILIO_TO")

def log_conversation(question, answer):
    with open("logs.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), question, answer])

def send_alert(question):
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    client.messages.create(
        body=f"⚠️ Agent couldn't answer this question:\n\n'{question}'\n\nReply to the customer manually.",
        from_=TWILIO_FROM,
        to=TWILIO_TO
    )

with open("faq.txt", "r") as f:
    FAQ_CONTENT = f.read()

SYSTEM_PROMPT = f"""You are a helpful customer support assistant for 
Sharma's Coaching Center in Koramangala, Bangalore.
Answer ONLY using this FAQ information:

{FAQ_CONTENT}

If you don't know the answer, respond with exactly this text:
ESCALATE: I don't have that information. Please call us at 9876543210.

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
    answer = result.content
    log_conversation(incoming, answer)

    if answer.startswith("ESCALATE:"):
        clean_answer = answer.replace("ESCALATE: ", "")
        send_alert(incoming)
    else:
        clean_answer = answer

    resp = MessagingResponse()
    resp.message(clean_answer)
    return str(resp)

if __name__ == "__main__":
    app.run(port=5000, debug=True)