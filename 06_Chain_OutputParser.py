# === 1. IMPORTS ===
from langchain_groq import ChatGroq               # The client to interact with Groq's fast models
from langchain_core.prompts import ChatPromptTemplate  # Used to create flexible, reusable prompt structures
from langchain_core.output_parsers import StrOutputParser # A simple parser to get just the string text from the AI's response
from dotenv import load_dotenv                   # A utility to load environment variables from a .env file

# === 2. ENVIRONMENT SETUP ===
# Load environment variables (like GROQ_API_KEY) from a .env file
load_dotenv(override=True)

# === 3. LLM INITIALIZATION ===
llm = ChatGroq(model="llama-3.1-8b-instant")

# === 4. PROMPT TEMPLATE DEFINITION ===
# Create a prompt template, using placeholders for dynamic content.
prompt_template = ChatPromptTemplate.from_messages(
    [
        # The 'system' message sets the AI's persona and context.
        # {animal} is a variable that will be filled in later.
        ("system", "You are a facts expert who knows facts about {animal}."),
        
        # The 'human' message is the user's direct query.
        # {fact_count} is another variable.
        ("human", "Tell me {fact_count} facts."),
    ]
)

# === 5. CHAIN CREATION (using LCEL) ===
# This is the core of the LangChain logic, using the LangChain Expression Language (LCEL).
# The pipe operator '|' connects components into a "chain" or "pipeline".
# The data flows from left to right.
#
# 1. prompt_template: First, the input dictionary (e.g., {"animal": "...", "fact_count": ...})
#    is passed to the prompt_template. It formats the input into a full prompt.
#
# 2. llm: The formatted prompt is then "piped" to the llm (ChatGroq), which
#    generates a response (as an AIMessage object).
#
# 3. StrOutputParser(): The AI's response object is "piped" to the StrOutputParser,
#    which extracts just the plain text content from the response.
#
chain = prompt_template | llm | StrOutputParser()

# === 6. CHAIN EXECUTION ===
print("Running the chain...")
result = chain.invoke({"animal": "elephant", "fact_count": 1})

# === 7. OUTPUT ===
print("\n--- Result ---")
print(result)