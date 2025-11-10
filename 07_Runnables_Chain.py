# === 1. IMPORTS ===
from langchain_groq import ChatGroq               # The client to interact with Groq's fast models
from langchain_core.prompts import ChatPromptTemplate  # Used to create flexible, reusable prompt structures
from langchain_core.runnables import RunnableLambda, RunnableSequence
from dotenv import load_dotenv                   # A utility to load environment variables from a .env file

# === 2. ENVIRONMENT SETUP ===
# Load environment variables (like GROQ_API_KEY) from a .env file
load_dotenv(override=True)

# === 3. LLM INITIALIZATION ===
model = ChatGroq(model="llama-3.1-8b-instant")

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

# Create individual runnables (steps in the chain)
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

# Create the RunnableSequence (equivalent to the LCEL chain)
chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)
# or chain = format_prompt | invoke_model | parse_output

# === 6. CHAIN EXECUTION ===
print("Running the chain...")
response = chain.invoke({"animal": "cat", "fact_count": 2})

# === 7. OUTPUT ===
print("\n--- Result ---")
print(response)