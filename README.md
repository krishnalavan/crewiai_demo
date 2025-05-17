# crewiai_demo
This is repo for demo purpose

Add API Key for openAI

Create a .env file and add the following line to it:
OPENAI_API_KEY=your_api_key_here
If you are using a groq or any other LLM, you can add the following line
GROQ_API_KEY=your_api_key_here
Generic : <LLLM_NAME>_API_KEY=your_api_key_here

Based on the LLM used, update the python file. For more information on LLM's, please refer the below link:
https://docs.crewai.com/concepts/llms

# Steps to execute the crew
After updating the OpenAI key,
In the browsed location >> run the below command
python bank_loan_single_agent.py
If your have installed python3, then run the below command
python3 bank_loan_single_agent.py
