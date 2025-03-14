import os
import json
import datetime
import csv
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

file_path = os.path.abspath("C:\\Users\\kani\\Desktop\\Chatbot\\intents.json")
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File {file_path} not found!")

with open(file_path, "r") as file:
    intents = json.load(file)

vectorizer = TfidfVectorizer()
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

x = vectorizer.fit_transform(patterns)

x_train, x_test, y_train, y_test = train_test_split(x, tags, test_size=0.2, random_state=42)


clf = RandomForestClassifier(n_estimators=100, random_state=42) 
clf.fit(x_train, y_train)


y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)


def chatbot(input_text):
    transformed_text = vectorizer.transform([input_text])
    predicted_tag = clf.predict(transformed_text)[0]
    for intent in intents:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses'])
    return "I'm sorry, I don't understand."


def main():
    st.title("Chatbot using Random Forest")

    menu = ["Home", "Conversation History", "About", "Model Accuracy"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Type a message below:")

        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        user_input = st.text_input("You:")
        if user_input:
            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120)

            
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv.writer(csvfile).writerow([user_input, response, timestamp])

    elif choice == "Conversation History":
        st.header("Conversation History")
        if os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  
                for row in csv_reader:
                    st.text(f"User: {row[0]} | Chatbot: {row[1]} | Time: {row[2]}")
        else:
            st.write("No conversation history found.")

    elif choice == "About":
        st.write("This chatbot is built using NLP techniques and a Random Forest classifier.")

    elif choice == "Model Accuracy":
        st.header("Model Accuracy")
        st.write(f"The accuracy of the Random Forest model on the test set is: **{accuracy * 100:.2f}%**")

if __name__ == '__main__':
    main()