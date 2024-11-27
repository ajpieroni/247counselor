# counselor.py

# Standard Libraries
import os
import logging
from typing import List
from typing_extensions import TypedDict

# Enable logging (optional)
logging.basicConfig(level=logging.DEBUG)

print("Starting counselor.py script...")

# LangChain Dependencies
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.chat_models import ChatOllama
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from langchain.schema import HumanMessage

# Define the LLM
local_llm = 'llama2'
print("Initializing LLM model...")
try:
    llama2 = ChatOllama(model=local_llm, temperature=0)
    print("LLM model initialized.")
except Exception as e:
    print(f"Error initializing LLM model: {e}")

# Categories and subcategories
CATEGORIES = {
    "Academics": ["Courses", "Standardized Testing", "Gap Analysis"],
    "ECA": ["ECA Recommendations", "Summer Programs", "Impact Metrics"],
    "Personal Development": ["Self-Reflection", "Growth Comparisons"],
    "College Applications": ["Essay Guidance", "Application Tracker", "College List", "Scholarships"],
}

# Define the GraphState
class GraphState(TypedDict):
    user_message: str
    category: str
    subcategory: str
    response: str
    data_check: bool
    probes: List[str]
    recommendations: List[str]
    feedback: str

# Define Pydantic models for output parsing
class RouterOutput(BaseModel):
    category: str = Field(..., description="The main category selected")
    subcategory: str = Field(None, description="The subcategory selected")

class ProbeOutput(BaseModel):
    probes: List[str] = Field(..., description="List of probing questions")

class PrerequisiteOutput(BaseModel):
    data_exists: bool = Field(..., description="Whether the required data exists")

class RecommendationOutput(BaseModel):
    recommendations: List[str] = Field(..., description="List of recommendations")

# Create parsers
router_parser = PydanticOutputParser(pydantic_object=RouterOutput)
probe_parser = PydanticOutputParser(pydantic_object=ProbeOutput)
prerequisite_parser = PydanticOutputParser(pydantic_object=PrerequisiteOutput)
recommendation_parser = PydanticOutputParser(pydantic_object=RecommendationOutput)

# Counselor response generation prompt
generate_prompt = PromptTemplate(
    template="""
You are KYROS, a compassionate and knowledgeable AI counselor available 24/7. Provide thoughtful, empathetic, and helpful guidance in the area of {category}. Ensure that your responses are supportive and promote well-being.

User's Message: {user_message}
""",
    input_variables=["user_message", "category"],
)

# Routing prompt
router_prompt = PromptTemplate(
    template="""
You are an assistant that categorizes user messages.

Determine the appropriate category and subcategory for the user's message from the following options:

Categories:
{categories}

Provide your output in JSON format matching the following schema:
{format_instructions}

User's Message: {user_message}
""",
    input_variables=["user_message", "categories"],
    partial_variables={"format_instructions": router_parser.get_format_instructions()},
)

# Prerequisite check prompt
prerequisite_prompt = PromptTemplate(
    template="""
Check if the necessary data exists in the user's profile to provide recommendations in the category of {category} and subcategory {subcategory}.

Provide your output in JSON format matching the following schema:
{format_instructions}
""",
    input_variables=["category", "subcategory"],
    partial_variables={"format_instructions": prerequisite_parser.get_format_instructions()},
)

# Probing prompt
probe_prompt = PromptTemplate(
    template="""
List at least four targeted questions to gather missing data necessary for providing recommendations in the category of {category}.

Provide your output in JSON format matching the following schema:
{format_instructions}
""",
    input_variables=["category"],
    partial_variables={"format_instructions": probe_parser.get_format_instructions()},
)

# Define functions for each node

def select_category(state):
    print("Step: Selecting Category")
    user_message = state['user_message']
    categories_str = "\n".join(
        [f"- {cat}: {', '.join(subs)}" for cat, subs in CATEGORIES.items()]
    )
    chain_input = {"user_message": user_message, "categories": categories_str}
    prompt = router_prompt.format(**chain_input)
    print("Prompt Sent to LLM:")
    print(prompt)
    messages = [HumanMessage(content=prompt)]
    response = llama2(messages)
    output_text = response.content
    print("LLM Output:")
    print(output_text)
    try:
        output = router_parser.parse(output_text)
    except Exception as e:
        print(f"Parsing error: {e}")
        print("LLM Output was not in the expected format.")
        return {}
    category = output.category or 'Academics'
    subcategory = output.subcategory or None
    return {"category": category, "subcategory": subcategory}

def prerequisite_check(state):
    print("Step: Performing Prerequisite Check")
    category = state['category']
    subcategory = state.get('subcategory', '')
    chain_input = {"category": category, "subcategory": subcategory}
    prompt = prerequisite_prompt.format(**chain_input)
    print("Prompt Sent to LLM:")
    print(prompt)
    messages = [HumanMessage(content=prompt)]
    response = llama2(messages)
    output_text = response.content
    print("LLM Output:")
    print(output_text)
    try:
        output = prerequisite_parser.parse(output_text)
    except Exception as e:
        print(f"Parsing error: {e}")
        print("LLM Output was not in the expected format.")
        return {}
    data_exists = output.data_exists
    return {"data_check": data_exists}

def inform(state):
    print("Step: Informing User")
    # Fetch known data from user profile (placeholder)
    known_data = "You have a GPA of 3.6 with strengths in Math and Science."
    response = f"{known_data}"
    return {"response": response}

def probe(state):
    print("Step: Probing for Missing Data")
    category = state['category']
    chain_input = {"category": category}
    prompt = probe_prompt.format(**chain_input)
    print("Prompt Sent to LLM:")
    print(prompt)
    messages = [HumanMessage(content=prompt)]
    response = llama2(messages)
    output_text = response.content
    print("LLM Output:")
    print(output_text)
    try:
        output = probe_parser.parse(output_text)
    except Exception as e:
        print(f"Parsing error: {e}")
        print("LLM Output was not in the expected format.")
        return {}
    probes = output.probes
    return {"probes": probes}

def recommend(state):
    print("Step: Generating Recommendations")
    user_message = state['user_message']
    category = state['category']
    chain_input = {"user_message": user_message, "category": category}
    prompt = generate_prompt.format(**chain_input)
    print("Prompt Sent to LLM:")
    print(prompt)
    messages = [HumanMessage(content=prompt)]
    response = llama2(messages)
    output_text = response.content
    print("LLM Output:")
    print(output_text)
    recommendations = [output_text.strip()]
    return {"recommendations": recommendations}

def get_feedback(state):
    print("Step: Collecting Feedback")
    # Placeholder: Simulate user feedback
    feedback = "Yes, these recommendations are helpful."
    return {"feedback": feedback}

def update_profile(state):
    print("Step: Updating User Profile")
    # Implement logic to update the profile (placeholder)
    return {}

# Build the state-based workflow using StateGraph

# Initialize the workflow
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("select_category", select_category)
workflow.add_node("prerequisite_check", prerequisite_check)
workflow.add_node("inform", inform)
workflow.add_node("probe", probe)
workflow.add_node("recommend", recommend)
workflow.add_node("get_feedback", get_feedback)
workflow.add_node("update_profile", update_profile)

# Define edges
workflow.set_entry_point("select_category")
workflow.add_edge("select_category", "prerequisite_check")
workflow.add_edge("prerequisite_check", "inform")
workflow.add_edge("inform", "probe")
workflow.add_edge("probe", "recommend")
workflow.add_edge("recommend", "get_feedback")
workflow.add_edge("get_feedback", "update_profile")
workflow.add_edge("update_profile", END)

# Compile the workflow
kyros_agent = workflow.compile()

# Define the function to run the agent
def run_kyros_counselor(user_message):
    output = kyros_agent.invoke({"user_message": user_message})
    print("=======")

    # Display the conversation
    if "response" in output:
        print(f"Inform: {output['response']}\n")
    if "probes" in output:
        print("Probing Questions:")
        for idx, probe in enumerate(output['probes'], 1):
            print(f"{idx}. {probe}")
            # Simulate user responses (you can replace this with actual input)
            user_input = input(f"Your answer to question {idx}: ")
    if "recommendations" in output:
        print("\nRecommendations:")
        for rec in output['recommendations']:
            print(f"- {rec}")
    if "feedback" in output:
        print(f"\nFeedback Received: {output['feedback']}")

print("Reached the main guard.")

if __name__ == "__main__":
    try:
        print("Inside main guard.")
        user_input = input("How can I assist you today? ")
        run_kyros_counselor(user_input)
    except Exception as e:
        print(f"An error occurred: {e}")