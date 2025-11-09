# --- 1. Imports ---
from dotenv import load_dotenv
from google.cloud import firestore  # Google's official Firestore library
from langchain_google_firestore import FirestoreChatMessageHistory  # LangChain's adapter for Firestore
from langchain_openai import ChatOpenAI  # LangChain's adapter for OpenAI

"""
Steps to replicate this example:
Before this code can run, you must complete several setup steps. The comments in your script correctly outline them:

1.Firebase Account: You need a Firebase account and a new project.
2.Firestore Database: You must activate the Firestore database within your Firebase project.
3.Project ID: You need to find and copy your unique PROJECT_ID from the Firebase console.
4.Google Cloud CLI: You must install the gcloud CLI tool. This is how your local computer gets permission to write to your cloud database.
5.Authentication: You need to authenticate the CLI by running gcloud init and gcloud auth application-default login in your terminal.
"""

# --- 2. Configuration ---
load_dotenv(override=True)

# --- Firebase Firestore Configuration ---
# Replace this with your own Project ID from the Firebase console
PROJECT_ID = "studio-588774631-fcaa8" 

# This is the unique key for a single conversation.
# Change this ID to start a new, separate conversation.
SESSION_ID = "user_session_new"  

# This is the name of the main collection in Firestore where chats are stored.
COLLECTION_NAME = "chat_history"

# --- 3. Initialize Clients and History ---

# Initialize the official Firestore client
# This uses your gcloud CLI authentication to securely connect.
print("Initializing Firestore Client...")
client = firestore.Client(project=PROJECT_ID)

# Initialize the LangChain chat history object
# This object links our specific SESSION_ID to our Firestore collection.
print("Initializing Firestore Chat Message History...")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client,
)
print("Chat History Initialized.")

# When we access .messages, it automatically fetches all existing messages
# from Firestore for this SESSION_ID. This is how memory is loaded.
print(f"Loaded {len(chat_history.messages)} previous messages for this session.")
print("---")

# --- 4. Initialize the Language Model ---
model = ChatOpenAI()

# --- 5. Start the Interactive Chat Loop ---
print("Start chatting with the AI. Type 'exit' to quit.")

while True:
    # 1. Get input from the user
    human_input = input("User: ")
    if human_input.lower() == "exit":
        break

    # 2. Add the user's message to the history
    # This action saves the message to the Firestore database.
    chat_history.add_user_message(human_input)
    # 3. Invoke the model with the *entire* conversation history
    # The .messages attribute contains all past user and AI messages.
    ai_response = model.invoke(chat_history.messages)
    # 4. Add the AI's response to the history
    # This also saves the AI's message to the Firestore database.
    chat_history.add_ai_message(ai_response.content)
    # 5. Print the AI's response to the console
    print(f"AI: {ai_response.content}")

print("Chat session ended.")