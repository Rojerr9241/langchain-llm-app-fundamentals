# Your code goes here
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()


prompt_template_str = """
Your task is to explain the concept of {concept} to me in a way that is:

1. Clear and intuitive
2. Concise (in under 100 words)
3. Tailored specifically to me and what I already know

Use the following information about me to personalize your explanation:

- Role: Data Analyst working toward AI Engineer
- Industry: Technology
- Background: AWS Certified Cloud & AI Practitioner, familiar with RAG architectures, LLMs, and sentiment analysis pipelines
- Goals: Building RAG systems and autonomous agents

The personalization should be subtle and natural. Avoid forced references to my background that don't genuinely enhance understanding of the concept.
"""

# Create a prompt template
prompt_template = PromptTemplate.from_template(prompt_template_str)

# Define the input variable
concept = "few-shot prompting"

# Format the prompt with the input variable
prompt = prompt_template.format(concept=concept)

print(prompt)

# Create a model interface
# model = init_chat_model("gpt-4o-mini", model_provider="openai")
model = init_chat_model("claude-sonnet-5", model_provider="anthropic")

# Call the model with the prompt
response = model.invoke(prompt)

# Print the generated content
print(response.text)

