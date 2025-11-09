# === 1. IMPORTS ===
# Import the specific chat models from their LangChain packages
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

# Import the message types we'll use to build our prompt
from langchain_core.messages import SystemMessage, HumanMessage

# Import the function to load API keys from our .env file
from dotenv import load_dotenv

# === 2. ENVIRONMENT SETUP ===
# Load environment variables from a .env file (like OPENAI_API_KEY,
# GOOGLE_API_KEY, and ANTHROPIC_API_KEY) into the environment.
# This is necessary for the models to authenticate.
load_dotenv(override=True)

# === 3. MODEL INITIALIZATION ===
# Create an instance for each LLM provider.
# LangChain provides a uniform interface, so we can use .invoke() on all of them.
print("Initializing models...")
llm_1 = ChatOpenAI(model="gpt-4o-mini")
llm_2 = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
llm_3 = ChatAnthropic(model="claude-3-haiku-20240307")

# === 4. PROMPT DEFINITION ===
# Define the "prompt" as a list of messages. This format is standard
# for chat models as it mimics a conversation history.
messages = [
    # SystemMessage: Sets the AI's persona, rules, or context.
    SystemMessage("Your are an expert in Social media content strategy"),
    
    # HumanMessage: Represents the user's direct question or input.
    HumanMessage("Give a short tip to crate a engaging post on Instragram")
]

# === 5. MODEL INVOCATION ===
# Send the same prompt (message list) to each model using .invoke()
# .invoke() runs the model and waits for the complete response.
print("Invoking models... (this may take a moment)")
result_1 = llm_1.invoke(messages)
result_2 = llm_2.invoke(messages)
result_3 = llm_3.invoke(messages)
print("...All models invoked.\n")

# === 6. DISPLAY RESULTS ===
# The 'result' objects are AIMessage objects.
# We access the actual text string using the .content attribute.

print("--- OpenAI (gpt-4o-mini) Response ---")
print(result_1.content)
print("\n" + "="*40 + "\n")

print("--- Google (gemini-1.5-flash) Response ---")
print(result_2.content)
print("\n" + "="*40 + "\n")

print("--- Anthropic (claude-3-haiku) Response ---")
print(result_3.content)
print("\n" + "="*40 + "\n")