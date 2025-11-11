import datetime
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain import hub

# --- 1. Load Environment Variables ---
# Make sure you have GROQ_API_KEY and TAVILY_API_KEY in your .env file
load_dotenv()

# --- 2. Define Tools ---

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """
    Returns the current LOCAL date and time of the computer running this code.
    """
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time


# List of all tools the agent can use
tools = [get_system_time]

# --- 3. Initialize Model and Prompt ---
llm = ChatGroq(model="qwen/qwen3-32b")

# Pull the standard ReAct agent prompt
prompt_template = hub.pull("hwchase17/react")

# --- 4. Create the Agent ---
agent = create_react_agent(llm, tools, prompt_template)

# The AgentExecutor is what runs the agent's "thought loop"
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True  # Set to True to see the agent's thoughts
)

# --- 5. Run the Agent ---
print("🚀 Running agent... \n")

query = "What is the current time in London? (You are in India). Just show the current time and not the date"

# We use invoke to run the agent with our query
try:
    response = agent_executor.invoke({"input": query})
    print("\n✅ Agent finished.")
    print("\nFinal Answer:")
    print(response["output"])

except Exception as e:
    print(f"\n❌ An error occurred: {e}")