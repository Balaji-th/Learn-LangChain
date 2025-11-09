"""
This script demonstrates two primary ways to create and use prompt templates
with LangChain, specifically using Google's Generative AI (Gemini).

1.  Using `ChatPromptTemplate.from_template`: Ideal for simple prompts
    where you fill in placeholders within a single string.
2.  Using `ChatPromptTemplate.from_messages`: Ideal for chat-style
    prompts that have distinct roles (like 'system' and 'human').
"""

# --- Imports ---
# Import the specific chat model we want to use (Gemini)
from langchain_google_genai import ChatGoogleGenerativeAI
# Import the core class for creating chat-based prompt templates
from langchain_core.prompts import ChatPromptTemplate
# Import the function to load environment variables (like API keys)
from dotenv import load_dotenv

# --- Setup ---
# Load environment variables from a .env file in the same directory.
# This is crucial for securely loading your GOOGLE_API_KEY.
# override=True means it will overwrite any existing system variables.
load_dotenv(override=True)

# Initialize the Large Language Model (LLM)
# We are using Google's "gemini-1.5-flash" model here.
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# =============================================================================
# Example 1: Prompt with Placeholders (using .from_template)
# =============================================================================
print("--- Example 1: Email Generation ---")

# 1. Define the template string with placeholders in curly braces {}
template = """
Write a {tone} mail to {company} expressing my interest in the {position} position.
Mention {skills} as a key strength.
Keep it to 4 lines.
"""

# 2. Create a prompt template object from the string
prompt_template_one = ChatPromptTemplate.from_template(template)

# 3. Create a concrete prompt by "invoking" the template with a dictionary.
#    The keys in the dictionary MUST match the placeholder names in the template.
prompt_one = prompt_template_one.invoke({
    "tone": "professional",
    "company": "Deloitte",
    "position": "AI Engineer",
    "skills": "Python, ML, and Gen AI"
})

# 4. Send the filled-in prompt to the model
result_one = llm.invoke(prompt_one)

# 5. Print the model's response
print(result_one.content)
print("-" * 30, "\n")


# =============================================================================
# Example 2: Prompt with System and Human Messages (using .from_messages)
# =============================================================================
print("--- Example 2: Comedian Jokes ---")

# 1. Define the chat messages as a list of tuples.
#    Each tuple is (role, content).
#    'system' sets the context or persona for the AI.
#    'human' is the user's input.
#    Placeholders {topic} and {joke_count} can be used in any message.
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]

# 2. Create a prompt template object from the list of messages
prompt_template_two = ChatPromptTemplate.from_messages(messages)

# 3. Create a concrete prompt by "invoking" the template with values
prompt_two = prompt_template_two.invoke({
    "topic": "lawyers",
    "joke_count": 3
})

# 4. Send the structured chat prompt to the model
result_two = llm.invoke(prompt_two)

# 5. Print the model's response
print(result_two.content)
print("-" * 30)