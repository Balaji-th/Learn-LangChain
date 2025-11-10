import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq 

# --- 1. Setup and Initialization ---
# Load environment variables from a .env file (like your OPENAI_API_KEY)
load_dotenv(override=True)

# --- Model Initialization ---
model = ChatGroq(model="llama-3.1-8b-instant")

# --- 2. Define Prompt Templates and Helper Functions ---

# Prompt template for the *first* step: generating a movie summary
summary_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie critic."),
        ("human", "Provide a brief summary of the movie {movie_name}."),
    ]
)

# A helper function to create the prompt for the plot analysis branch.
def analyze_plot(plot_summary):
    plot_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a movie critic."),
            # This prompt asks the AI to analyze the *summary* it just wrote
            ("human", "Analyze the plot from this summary: {plot}. What are its strengths and weaknesses?"),
        ]
    )
    # Format the prompt with the received summary
    return plot_template.format_prompt(plot=plot_summary)

# A helper function to create the prompt for the character analysis branch.
def analyze_characters(character_summary):
    character_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a movie critic."),
            # This prompt asks the AI to analyze the characters *based on the summary*
            ("human", "Analyze the characters from this summary: {characters}. What are their strengths and weaknesses?"),
        ]
    )
    # Format the prompt with the received summary
    return character_template.format_prompt(characters=character_summary)

# A final helper function to combine the results from the two parallel branches
def combine_verdicts(plot_analysis, character_analysis):
    return f"Plot Analysis (from summary):\n{plot_analysis}\n\nCharacter Analysis (from summary):\n{character_analysis}"

# --- 3. Define Parallel Branches using LCEL ---

# This chain will be used for the "plot" branch
# 1. RunnableLambda: Takes the input (the movie summary) and uses our helper function to create a new prompt.
# 2. model: Sends that new prompt to the LLM.
# 3. StrOutputParser: Parses the LLM's message into a simple string.
plot_branch_chain = (
    RunnableLambda(lambda x: analyze_plot(x)) 
    | model 
    | StrOutputParser()
)

# This chain will be used for the "characters" branch
# It follows the exact same logic as the plot branch
character_branch_chain = (
    RunnableLambda(lambda x: analyze_characters(x)) 
    | model 
    | StrOutputParser()
)

# --- 4. Define the Main Chain ---

# This is where all the pieces are combined using the | (pipe) operator
chain = (
    summary_template    # Step 1: Start with the summary template. This will take {"movie_name": "Inception"} as input.
    | model             # Step 2: Send the formatted prompt to the model.
    | StrOutputParser() # Step 3: Parse the model's output into a string (this is the movie summary).
    | RunnableParallel(
        branches={
            "plot": plot_branch_chain,
            "characters": character_branch_chain,
        }
    )                   # Step 4: Use RunnableParallel to run multiple chains *at the same time*.
    | RunnableLambda(lambda x: combine_verdicts(x["branches"]["plot"], x["branches"]["characters"]))
)                       # Step 5: The output of RunnableParallel is a dictionary, e.g.: # {'branches': {'plot': '...', 'characters': '...'}}

# --- 5. Run the Chain ---
print("--- Running LCEL Chain for 'Inception' ---")
result = chain.invoke({"movie_name": "Inception"})

# Print the final combined result
print(result)