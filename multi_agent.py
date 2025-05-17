import os
from openai import OpenAI
from dotenv import load_dotenv
from groq import Groq
from crewai import Crew, Agent, Task
from crewai import LLM
load_dotenv()

# Load environment variables from .env file
api_key = os.getenv("GROQ_API_KEY")
print(f"API Key: {api_key}")    

groq_client = Groq(api_key=api_key) 

# Initialize LLM with the Groq client
llm = LLM(groq_client=groq_client,
          model="gemma2-9b-it",
          temperature=0.7)

# Initialize Agents

researcher_agent = Agent(
    name="Researcher",
    description="An AI agent that researches and gathers information about the latest AI tools/frameworks.",
    goal="To research, gather and summarize information about the latest AI tools/frameworks.",
    role="Researcher",  
    backstory="You are an AI researcher who is passionate about exploring new technologies. You have been tasked to research and gather information about the latest AI tools/frameworks."      
)

writer_agent = Agent(
    name="Writer",
    description="An AI agent that can write a blog on any latest content about using AI in the banking sector.",
    goal="To write a blog on using AI in banking sector.",
    role="Writer",
    backstory="You are a write who can research about all latest AI tools/framework and write blog on how effectively those tools can be used in Banking sector."
)

reviewer_agent = Agent(
    name="Reviewer",
    description="An AI agent that can review the blog written by the Writer agent.",
    goal="To review the blog and provide feedback.",
    role="Reviewer",
    backstory="You are a reviewer who can review the blog and provide feedback."
)

# Initialize Tasks

gather_information_task = Task( 
    name="Gather Information",
    description="Research and gather information about the latest AI tools/frameworks.",
    agent=researcher_agent,
    expected_output="A summary of the research findings.",
)

write_blog_task = Task(
    name="Write Blog",
    description="Write a blog on using AI in banking sector.",
    agent=writer_agent,
    expected_output="A blog on using AI in banking sector.",
)

review_blog_task = Task(
    name="Review Blog",
    description="Review the blog and provide feedback.",
    agent=reviewer_agent,
    expected_output="Feedback on the blog.",
)
# Initialize Crew
crew = Crew(
    agents=[researcher_agent, writer_agent, reviewer_agent],
    tasks=[gather_information_task, write_blog_task, review_blog_task]
)
result = crew.kickoff()
print(result)