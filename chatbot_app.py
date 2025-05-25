# chatbot_app.py
import subprocess
import sys

# Failsafe install of nltk
try:
    import nltk
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
    import nltk


import os
import ssl
import nltk
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Ensure nltk downloads work
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')

# Define intents
intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
        "responses": ["Hi there", "Hello", "Hey", "I'm fine", "Thank you", "Nothing much"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
        "responses": ["Goodbye", "See you later", "Take care"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome", "No problem", "Glad I could help"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
        "responses": ["I am a chatbot", "My purpose is to assist you", "I can answer questions and provide assistance"]
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
        "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?", "How can I assist you?"]
    },
    {
        "tag": "age",
        "patterns": ["How old are you?", "What's your goal?"],
        "responses": ["I don't have an age. I'm a chatbot", "I was just born in the digital world", "Age is just a number for me."]
    },
    {
        "tag": "weather",
        "patterns": ["What's the weather like", "How's the weather today?"],
        "responses": ["I'm sorry, I cannot provide real-time weather information.", "You can check the weather on a weather app or website"]
    },
    {
        "tag": "budget",
        "patterns": ["How can I make a budget?", "What's a good budgeting strategy", "How do I create a budget"],
        "responses": [
            "To make a budget, start by tracking your income and expenses. Then allocate your income toward essentials, savings, and discretionary items.",
            "A good budgeting strategy is the 50/30/20 rule: 50% essentials, 30% discretionary, 20% savings.",
            "Start by setting financial goals, then track spending and assign limits."
        ]
    },
    {
        "tag": "credit_score",
        "patterns": ["What is a credit score?", "How to improve credit score", "Why is credit score important"],
        "responses": [
            "A credit score is a number that represents your creditworthiness.",
            "To improve your credit score, pay bills on time and keep credit usage low.",
            "Credit scores impact your ability to get loans and interest rates."
        ]
    }
]

# Prepare training data
corpus = []
tags = []

for intent in intents:
    for pattern in intent["patterns"]:
        corpus.append(pattern.lower())
        tags.append(intent["tag"])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

model = LogisticRegression()
model.fit(X, tags)

# Define chatbot response function
def chatbot_response(user_input):
    user_input = user_input.lower()
    X_test = vectorizer.transform([user_input])
    predicted_tag = model.predict(X_test)[0]

    for intent in intents:
        if intent["tag"] == predicted_tag:
            return random.choice(intent["responses"])

    return "Sorry, I don't understand that."

# Streamlit UI
def main():
    st.title("Chatbot")
    st.write("Type your message below to start chatting:")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("You:")

    if user_input:
        response = chatbot_response(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))

    for speaker, message in st.session_state.chat_history:
        st.write(f"**{speaker}:** {message}")

if __name__ == "__main__":
    main()
