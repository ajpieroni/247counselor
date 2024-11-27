# counselor.py

# Standard Libraries
import os
import logging
import json
from typing import List
from typing_extensions import TypedDict

# Enable logging (optional)
logging.basicConfig(level=logging.INFO)

print("Starting counselor.py script...")

# LangChain Dependencies
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOllama
from langchain.schema import HumanMessage
from langgraph.graph import END, StateGraph

# Define the LLM
local_llm = 'llama2'
print("Initializing LLM model...")
try:
    llama2 = ChatOllama(model=local_llm, temperature=0)
    print("LLM model initialized.")
except Exception as e:
    print(f"Error initializing LLM model: {e}")
    exit(1)

# Categories and subcategories
CATEGORIES = {
    "Academics": ["Courses", "Standardized Testing", "Gap Analysis"],
    "Extracurricular Activities": ["Clubs", "Sports", "Volunteer Work", "Leadership Opportunities"],
    "Personal Development": ["Self-Reflection", "Growth Comparisons"],
    "College Applications": ["Essay Guidance", "Application Tracker", "College List", "Scholarships"],
}

# Define the GraphState
class GraphState(TypedDict):
    user_context: str
    user_message: str
    category: str
    subcategory: str
    response: str
    data_check: bool
    probes: List[str]
    recommendations: List[str]
    feedback: str
    user_profile: dict  # To store user data

# Counselor response generation prompt
generate_prompt = PromptTemplate(
    template="""
You are an expert college counselor providing personalized advice to high school students. Offer tailored guidance based on the user's profile, including GPA, extracurricular activities, zipcode, and high school size. Focus on the category of {category} and subcategory of {subcategory}.

User's Context: {user_context}

User's Profile:
GPA: {gpa}
Extracurricular Activities: {extracurriculars}
Zipcode: {zipcode}
High School Size: {high_school_size}

User's Message: {user_message}
""",
    input_variables=["user_context", "user_message", "category", "subcategory", "gpa", "extracurriculars", "zipcode", "high_school_size"],
)

# Routing prompt
router_prompt = PromptTemplate(
    template="""
You are an assistant that categorizes user messages.

Determine the appropriate category and subcategory for the user's message from the following options:

Categories:
{categories}

Based on the user's message, select the most relevant category and subcategory.

Provide your output in JSON format as follows:
{{ "category": "CategoryName", "subcategory": "SubcategoryName" }}

If the subcategory is not specified, you can set it to null.

User's Message: {user_message}
""",
    input_variables=["user_message", "categories"],
)

# Prerequisite check prompt
prerequisite_prompt = PromptTemplate(
    template="""
Check if the necessary data exists in the user's profile to provide personalized recommendations in the category of {category} and subcategory {subcategory}. Specifically, check for the following base-level metrics:

- GPA
- Extracurricular Activities
- Zipcode
- High School Size

Respond with {{ "data_exists": true }} if all the data exists, or {{ "data_exists": false }} if any of it is missing.
""",
    input_variables=["category", "subcategory"],
)

# Probing prompt for missing base-level metrics
probe_prompt = PromptTemplate(
    template="""
Ask the user for the following missing information to provide personalized recommendations:

{missing_metrics}

Provide your output in JSON format as follows:
{{ "probes": ["Question 1", "Question 2", ...] }}
""",
    input_variables=["missing_metrics"],
)

# Define functions for each node

def select_category(state):
    print("\n===== Step: Selecting Category =====")
    user_message = state['user_message']
    categories_str = "\n".join(
        [f"- {cat}: {', '.join(subs)}" for cat, subs in CATEGORIES.items()]
    )
    chain_input = {"user_message": user_message, "categories": categories_str}
    prompt = router_prompt.format(**chain_input)
    messages = [HumanMessage(content=prompt)]
    response = llama2(messages)
    output_text = response.content.strip()
    try:
        output_json = json.loads(output_text)
        category = output_json.get('category', 'Academics')
        subcategory = output_json.get('subcategory')
        if subcategory in [None, '', 'null']:
            # Subcategory not specified, prompt the user
            print(f"\nYou have selected the category: {category}")
            # Get the list of subcategories for the selected category
            subcategories = CATEGORIES.get(category, [])
            if subcategories:
                print("Please select a subcategory from the following options:")
                for idx, sub in enumerate(subcategories, 1):
                    print(f"{idx}. {sub}")
                # Prompt the user to select a subcategory
                while True:
                    try:
                        selection = int(input("Enter the number of your choice: "))
                        if 1 <= selection <= len(subcategories):
                            subcategory = subcategories[selection - 1]
                            print(f"You have selected subcategory: {subcategory}")
                            break
                        else:
                            print("Invalid selection. Please try again.")
                    except ValueError:
                        print("Please enter a valid number.")
            else:
                print("No subcategories available for the selected category.")
        else:
            print(f"\nYou have selected the category: {category} and subcategory: {subcategory}")
        state['category'] = category
        state['subcategory'] = subcategory
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print("LLM Output was not in valid JSON format.")
        # Default to Academics category
        state['category'] = 'Academics'
        state['subcategory'] = None
    return state

def prerequisite_check(state):
    print("\n===== Step: Performing Prerequisite Check =====")
    # Simulate checking user's profile for base-level metrics
    user_profile = state.get('user_profile', {})
    required_metrics = ['gpa', 'extracurriculars', 'zipcode', 'high_school_size']
    missing_metrics = [metric for metric in required_metrics if metric not in user_profile]
    if missing_metrics:
        state['data_check'] = False
        state['missing_metrics'] = missing_metrics
    else:
        state['data_check'] = True
        state['missing_metrics'] = []
    return state

def inform_or_probe(state):
    data_check = state.get('data_check', False)
    if data_check:
        # Inform the user
        return inform(state)
    else:
        # Probe for missing base-level metrics
        return probe(state)

def inform(state):
    print("\n===== Step: Informing User =====")
    # Fetch known data from user profile
    user_profile = state.get('user_profile', {})
    known_data = f"GPA: {user_profile.get('gpa')}, Extracurricular Activities: {user_profile.get('extracurriculars')}, Zipcode: {user_profile.get('zipcode')}, High School Size: {user_profile.get('high_school_size')}."
    response = f"We have your profile information: {known_data}"
    print(f"\n{response}")
    state['response'] = response
    return state

def probe(state):
    print("\n===== Step: Collecting Necessary Information =====")
    missing_metrics = state.get('missing_metrics', [])
    user_profile = state.get('user_profile', {})
    # Ask user for missing base-level metrics
    for metric in missing_metrics:
        if metric == 'gpa':
            question = "Please enter your GPA (e.g., 3.8): "
        elif metric == 'extracurriculars':
            question = "Please list your extracurricular activities (e.g., Robotics Club, Soccer Team): "
        elif metric == 'zipcode':
            question = "Please provide your zipcode: "
        elif metric == 'high_school_size':
            question = "Please provide the size of your high school (number of students): "
        else:
            question = f"Please provide your {metric.replace('_', ' ')}: "
        answer = input(question)
        user_profile[metric] = answer
    state['user_profile'] = user_profile
    # After collecting missing data, set data_check to True
    state['data_check'] = True
    # Inform the user with the updated profile
    return inform(state)

def recommend(state):
    print("\n===== Step: Generating Personalized Recommendations =====")
    user_context = state['user_context']
    user_message = state['user_message']
    category = state['category']
    subcategory = state['subcategory']
    user_profile = state.get('user_profile', {})
    gpa = user_profile.get('gpa', 'N/A')
    extracurriculars = user_profile.get('extracurriculars', 'N/A')
    zipcode = user_profile.get('zipcode', 'N/A')
    high_school_size = user_profile.get('high_school_size', 'N/A')

    # For future RAG implementation:
    # Here is where you can integrate Retrieval-Augmented Generation (RAG) to enhance recommendations.
    # Steps to implement RAG:
    # 1. Use user's context, message, and profile to query a vector store of documents.
    # 2. Retrieve relevant documents based on similarity search.
    # 3. Include retrieved information in the prompt to the LLM.

    # For now, we'll fetch the LLM response without RAG.
    chain_input = {
        "user_context": user_context,
        "user_message": user_message,
        "category": category,
        "subcategory": subcategory,
        "gpa": gpa,
        "extracurriculars": extracurriculars,
        "zipcode": zipcode,
        "high_school_size": high_school_size,
    }
    prompt = generate_prompt.format(**chain_input)
    messages = [HumanMessage(content=prompt)]
    response = llama2(messages)
    output_text = response.content.strip()
    recommendations = [output_text]
    print("\nRecommendations:")
    print(f"{output_text}")
    state['recommendations'] = recommendations
    return state

def get_feedback(state):
    print("\n===== Step: Collecting Feedback =====")
    feedback = input("Was this information helpful? (Yes/No): ")
    state['feedback'] = feedback
    return state

def update_profile(state):
    print("\n===== Step: Updating User Profile =====")
    # Implement logic to update the profile (placeholder)
    print("Your profile has been updated with the new information.")
    return state

# Build the state-based workflow using StateGraph

# Initialize the workflow
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("select_category", select_category)
workflow.add_node("prerequisite_check", prerequisite_check)
workflow.add_node("inform_or_probe", inform_or_probe)
workflow.add_node("recommend", recommend)
workflow.add_node("get_feedback", get_feedback)
workflow.add_node("update_profile", update_profile)

# Define edges
workflow.set_entry_point("select_category")
workflow.add_edge("select_category", "prerequisite_check")
workflow.add_edge("prerequisite_check", "inform_or_probe")
workflow.add_edge("inform_or_probe", "recommend")
workflow.add_edge("recommend", "get_feedback")
workflow.add_edge("get_feedback", "update_profile")
workflow.add_edge("update_profile", END)

# Compile the workflow
kyros_agent = workflow.compile()

# Define the function to run the agent
def run_kyros_counselor():
    print("\n===== Welcome to KYROS AI Counselor =====")
    user_context = input("Please provide some context about yourself (e.g., 'I am a 10th-grade student interested in engineering'): ")
    user_message = input("\nHow can I assist you today?\nYour message: ")
    state = {
        "user_context": user_context,
        "user_message": user_message,
        "user_profile": {},  # Initialize an empty user profile
    }
    state = kyros_agent.invoke(state)
    print("\n===== Thank you for using KYROS AI Counselor! =====\n")

# Run the counselor if executed as a script
if __name__ == "__main__":
    try:
        run_kyros_counselor()
    except Exception as e:
        print(f"An error occurred: {e}")