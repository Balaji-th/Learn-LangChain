# 1. IMPORTS
# Import the specific class 'ChatOpenAI' from the 'langchain_openai' library.
from langchain_openai import ChatOpenAI

# Import the 'load_dotenv' function from the 'dotenv' library.
# This function is used to load environment variables from a .env file.
from dotenv import load_dotenv

# 2. CONFIGURATION
# Load environment variables from a .env file located in the same directory.
load_dotenv(override=True)

# 3. EXECUTION : Create an instance of the language model (LLM).
llm = ChatOpenAI(model="gpt-4o-mini")

# Define the prompt you want to send to the model.
prompt = "What is AI?"
# Use the .invoke() method to send the prompt to the AI model.
result = llm.invoke(prompt)
# Print the AI's answer.
print(result.content)