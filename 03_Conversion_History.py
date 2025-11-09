# --- 1. Imports ---
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# --- 2. Setup ---

# Load environment variables (like GOOGLE_API_KEY) from your .env file
# This makes your API keys available to the script without hard-coding them.
load_dotenv(override=True)

# Initialize the generative AI model
# We're using Google's Gemini-1.5-Flash model here.
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# --- 3. Chat History Management ---

# Create an empty list to store the entire conversation history.
# This list will hold all System, Human, and AI messages.
chat_history = [] 

# Set an initial system message to define the AI's role or persona.
# This message is sent to the model first to guide its behavior.
system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history.append(system_message) # Add the system message to the history

print("AI: Hello! How can I help you today? (Type 'exit' to end the chat)")

# --- 4. Main Chat Loop ---

while True:
    # Get input from the user via the command line
    query = input("You: ")
    
    # Check if the user wants to quit the chat
    if query.lower() == "exit":
        print("AI: Goodbye!")
        break
    
    # Add the user's message to the chat history
    # We wrap it in a HumanMessage object so the model knows who said it.
    chat_history.append(HumanMessage(content=query))
    
    # --- 5. Model Invocation ---
    
    # Send the *entire* chat_history list to the model.
    # This is how the model gets the context of the full conversation.
    result = model.invoke(chat_history)
    
    # Extract the text content from the model's response
    response = result.content
    
    # Add the AI's response back to the chat history
    # We wrap it in an AIMessage object. This is crucial for the model
    # to remember what *it* said in the next turn.
    chat_history.append(AIMessage(content=response))
    
    # Print the AI's response to the console
    print(f"AI: {response}")

# --- 6. Final Output ---

# After the loop breaks, print the entire message history
# This shows you the list of objects that was maintained.
print("\n---- Final Message History ----")
print(chat_history)