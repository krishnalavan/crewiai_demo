import os
from openai import OpenAI
from dotenv import load_dotenv
from crewai import Crew, Agent, Task
load_dotenv()

# Load environment variables from .env file
api_key = os.getenv("OPENAI_API_KEY")
print(f"API Key: {api_key}")

if not api_key:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=api_key)

response = client.responses.create(
    model="gpt-4o",
    instructions="You are a helpful assistant.",
    input="What is the capital of France?",
)

print(response.output_text)

# Initialize Agent
agent1 = Agent(
    name="Researcher",
    description="An AI agent that researches and gathers information,",
    goal="To research, gather and summarize information on a given topic.",
    role="Researcher",
    backstory="You are a researcher who gathers information."
)

# Initialize Task
research_task = Task(
    name="Research Topic",
    description="Research and gather information about a specific topic.",
    agent=agent1,
    expected_output="A summary of the research findings.",
)

# Initialize Crew
crew = Crew(agents=[agent1],tasks=[research_task])
crew.kickoff()

