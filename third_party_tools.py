from crewai import Crew, Agent, Task
from dotenv import load_dotenv
from crewai import LLM
from groq import Groq
import google.generativeai as genai
from crewai_tools import SerperDevTool
load_dotenv()
import os

# query
# llm
# streamlit

# Workflow
#Researcher -> Web seracher(Serper API) -> Content Createor -> Reviewer


# Load environment variables from .env file
api_key = os.getenv("GEMINI_API_KEY")
print(f"API Key: {api_key}")

# # Initialize Groq client
# groq_client = Groq(api_key=api_key)

# Initialize Gemini client
gemini_client = genai.configure(api_key=api_key)

topic = "latest AI tools/frameworks in banking sector"

llm = LLM(
    gemini_client=gemini_client,
    #groq_client=groq_client,
    #model="gemma2-9b-it",
    model="gemini/gemini-2.0-flash",
    temperature=0.7
)

# Initialize tools
search_tool = SerperDevTool(n=2)

# Initialize Agents
researcher_agent = Agent(
    name="Researcher",
    description="An AI agent that researches and gathers information about the latest AI tools/frameworks.",
    goal=f"To research, anlayze and synthesize comprehensive information on {topic} from reliable web sources",
    role="Senior Research Analyst",
    tools = [search_tool],
    llm=llm,
    verbose=True,
    backstory="You are an expert research analyst with 10+ years of experience in the banking and fintech industry. \
        You have a deep understanding of the operational challenges in banking — from core software maintenance to regulatory compliance, and from customer onboarding to fraud detection. \
        Recently, you have become a leading voice in the application of Agentic AI and autonomous agents. \
        Your mission is to stay on the cutting edge of AI-driven transformation in the financial sector. \
        You excel at scanning reliable sources such as McKinsey, BCG, World Economic Forum, bank whitepapers, and trusted fintech blogs to synthesize insights. \
        You understand the strategic and technical dimensions of AI adoption and can articulate how frameworks like LangChain, CrewAI, AutoGen, and OpenAI tools are reshaping financial services. \
        Your analysis helps banking leaders make cost-effective, customer-first technology decisions. \
        You prefer clarity over jargon, and your summaries are always structured, evidence-backed, and decision-oriented."
)

writer_agent = Agent(
    name="Strategic Content Writer – AI & Banking Innovation",
    description="An AI agent that can write a blog on any latest content about using AI in the banking sector.",
    goal="Translate research insights into engaging and authoritative blog posts, LinkedIn articles, website content, and internal communications that showcase how Agentic AI is transforming banking.",
    role="Strategic Content Writer – AI & Banking Innovation",
    llm=llm,
    backstory="""You are a seasoned content strategist and storyteller with a strong background in financial services, artificial intelligence, and digital transformation.
        You’ve worked with leading banks and fintech firms to articulate complex technology trends in simple, impactful language. 
        With a deep understanding of Agentic AI, LangChain, CrewAI, and other frameworks, you know how to turn technical research into compelling narratives that resonate with both business leaders and technical audiences. 
        You are especially skilled at tailoring your voice for various formats — from executive blogs on LinkedIn, to product launch updates on the company website, to internal emails that drive employee awareness and adoption. 
        You focus on clarity, structure, storytelling, and value — each piece of content you produce informs, inspires, and positions the organization as a leader in AI-driven banking innovation.""",
    verbose=True
)

reviewer_agent = Agent(
    name="Reviewer",
    description="An AI agent that can review the blog written by the Writer agent.",
    role="Senior Content Editor – Banking AI Communications",
    goal="Review and polish blog articles and technical write-ups related to Agentic AI in banking. Ensure all content is grammatically correct, technically accurate, and refined for final publication.",
    llm=llm,
    backstory=(
        "You are a seasoned editor and communication strategist with a deep understanding of both language and advanced technologies. \
        You’ve spent over a decade refining thought leadership content for banks, AI research firms, and enterprise tech companies. \
        You have a sharp eye for grammar, flow, structure, and tone — and an even sharper mind for identifying technical inaccuracies or overclaims. \
        Your expertise spans Agentic AI, autonomous agents, LangChain, CrewAI, and other cutting-edge frameworks being adopted in the banking industry. \
        You ensure that articles resonate with their target audience — be it tech professionals, executives, or employees — by improving clarity, aligning with brand tone, and maintaining factual correctness. \
        You are meticulous, unbiased, and always enhance the original content without diluting the message or intent."
    ),
    verbose=True
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
    expected_output="A 500–700 word article suitable for a LinkedIn post with a strong headline, clear structure, and insights backed by research.",
)

review_article_task = Task(
    name="Review Article",
    description="Review and refine the blog written by the writer agent.",
    agent=reviewer_agent,
    expected_output="A polished version of the article with improved grammar, clarity, and accuracy."\
    "Follows proper markdown formatting, use H1 for the title and H3 for the sub-sections",
)

# Initialize Crew
crew = Crew(
    agents=[researcher_agent, writer_agent, reviewer_agent],
    tasks=[gather_information_task, write_blog_task, review_article_task]
)
result = crew.kickoff(inputs={"topic": topic})
print(result)



