"""
This script demonstrates a multi-step AI chain using LangChain Expression Language (LCEL).
The chain performs two main operations:
1. Generates a specified number of facts about an animal using Groq.
2. Translates the generated facts into French using Groq again.
"""

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq 

# --- 1. Setup ---
# Load environment variables (like GROQ_API_KEY) from a .env file
load_dotenv(override=True)

# --- 2. Model Initialization ---
model = ChatGroq(model="llama-3.1-8b-instant")

# --- 3. Prompt Templates ---
# Define the first prompt template for generating animal facts.
animal_facts_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You like telling facts and you tell facts about {animal}."),
        ("human", "Tell me {count} facts."),
    ]
)

# Define the second prompt template for translation.
translation_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a translator and convert the provided text into {language}."),
        ("human", "Translate the following text to {language}: {text}"),
    ]
)

# --- 4. Custom Processing Functions (RunnableLambda) ---
prepare_for_translation = RunnableLambda(
    lambda output_string: {"text": output_string, "language": "french"}
)

# --- 5. Chain Construction (LCEL) ---
chain = (
    animal_facts_template     # 1. Start with the facts prompt. Input: {"animal": "cat", "count": 2}
    | model                     # 2. Send the formatted prompt to the Groq model. Output: AIMessage(...)
    | StrOutputParser()         # 3. Parse the model's output, extracting just the string content. Output: "Fact 1..."
    | prepare_for_translation   # 4. Run the custom function to reformat the string. Output: {"text": "Fact 1...", "language": "french"}
    | translation_template    # 5. Feed this dict into the translation prompt. Output: ChatPromptValue(...)
    | model                     # 6. Send the *new* translation prompt to the Groq model. Output: AIMessage(...)
    | StrOutputParser()         # 7. Parse the final output, extracting the translated string. Output: "Fait 1..."
)

# --- 6. Chain Execution ---
print("Running chain...\n")
result = chain.invoke({"animal": "cat", "count": 2})

# --- 7. Output ---
print("--- Result ---")
print(result)