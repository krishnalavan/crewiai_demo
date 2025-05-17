import os
from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv
from crewai import Crew, Agent, Task
from crewai import LLM
load_dotenv()

# Load environment variables from .env file
api_key = os.getenv("GROQ_API_KEY")
print(f"API Key: {api_key}")

if not api_key:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

client = Groq(api_key=api_key)

llm = LLM(
    model="gemma2-9b-it",
    temperature=0.7
    # instructions="You are a helpful assistant.",
    # input="What is the capital of India?",
)
response = client.chat.completions.create(
    model="gemma2-9b-it",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of India?"}
    ],
    temperature=0.7
)

print(response.choices[0].message.content)

# Initialize Agent
agent1 = Agent(
    name="Researcher",
    description="An AI agent that researches and gathers information about bank loans.",
    goal="To research, gather and summarize information information about bank loans.",
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
result =  crew.kickoff()
print(result)

