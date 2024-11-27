# counselor.py

# Standard Libraries
import os
import logging
import json
from typing import List, Dict
from typing_extensions import TypedDict

# Enable logging (optional)
logging.basicConfig(level=logging.INFO)

print("Starting counselor.py script...")

# LangChain Dependencies
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOllama
from langchain.schema import HumanMessage

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
    "Academics": ["Course Selection", "Standardized Testing", "Gap Analysis"],
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
    user_profile: Dict[str, str]  # To store user data
    additional_info: Dict[str, str]  # To store answers to additional probes

# Counselor response generation prompt
generate_prompt = PromptTemplate(
    template="""
You are an experienced and empathetic college counselor providing personalized advice to high school students. Offer tailored guidance based on the user's profile and additional information. Be as specific and personalized as possible, addressing the student's individual situation, goals, and challenges. Focus on the category of {category} and subcategory of {subcategory}.

User's Context: {user_context}

User's Profile:
GPA: {gpa}
Extracurricular Activities: {extracurriculars}
Zipcode: {zipcode}
High School Size: {high_school_size}

Additional Information:
{additional_info}

User's Message: {user_message}
""",
    input_variables=[
        "user_context",
        "user_message",
        "category",
        "subcategory",
        "gpa",
        "extracurriculars",
        "zipcode",
        "high_school_size",
        "additional_info",
    ],
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

# Define functions for each step

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
    print("\n===== Step: Checking Profile Information =====")
    # Simulate checking user's profile for base-level metrics
    user_profile = state.get('user_profile', {})
    required_metrics = ['gpa', 'extracurriculars', 'zipcode', 'high_school_size']
    missing_metrics = [metric for metric in required_metrics if not user_profile.get(metric)]
    if missing_metrics:
        state['data_check'] = False
        state['missing_metrics'] = missing_metrics
    else:
        state['data_check'] = True
        state['missing_metrics'] = []
    return state

def collect_profile_info(state):
    print("\n===== Step: Collecting Profile Information =====")
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
    return state

def inform(state):
    print("\n===== Step: Reviewing Your Profile =====")
    # Fetch known data from user profile
    user_profile = state.get('user_profile', {})
    known_data = f"GPA: {user_profile.get('gpa')}, Extracurricular Activities: {user_profile.get('extracurriculars')}, Zipcode: {user_profile.get('zipcode')}, High School Size: {user_profile.get('high_school_size')}."
    response = f"We have noted your profile information: {known_data}"
    print(f"\n{response}")
    state['response'] = response
    return state

def probe_for_details(state):
    print("\n===== Step: Gathering Additional Information =====")
    category = state['category']
    subcategory = state['subcategory']
    # Define additional questions based on category and subcategory
    additional_questions = []
    if category == 'Academics' and subcategory == 'Course Selection':
        additional_questions = [
            "Which science courses have you taken so far?",
            "Are you interested in Advanced Placement (AP) or International Baccalaureate (IB) courses?",
            "Do you have any specific colleges or programs in mind?",
            "What are your strengths and weaknesses in academics?",
        ]
    elif category == 'Extracurricular Activities' and subcategory == 'Leadership Opportunities':
        additional_questions = [
            "What leadership roles have you held in your extracurricular activities?",
            "Are there specific leadership positions you're aiming for?",
            "How do you think these roles will contribute to your future goals?",
            "What challenges have you faced in leadership roles before?",
        ]
    else:
        additional_questions = [
            "Please provide more details about your interests and goals.",
            "Are there any specific challenges you're facing?",
            "What do you hope to achieve in this area?",
            "Is there anything else you'd like to share?",
        ]
    # Collect user's answers
    additional_info = {}
    for question in additional_questions:
        answer = input(f"{question}\nYour answer: ")
        additional_info[question] = answer
    state['additional_info'] = additional_info
    return state

def recommend(state):
    print("\n===== Step: Generating Personalized Recommendations =====")
    user_context = state['user_context']
    user_message = state['user_message']
    category = state['category']
    subcategory = state['subcategory']
    user_profile = state.get('user_profile', {})
    additional_info = state.get('additional_info', {})
    gpa = user_profile.get('gpa', 'N/A')
    extracurriculars = user_profile.get('extracurriculars', 'N/A')
    zipcode = user_profile.get('zipcode', 'N/A')
    high_school_size = user_profile.get('high_school_size', 'N/A')
    additional_info_str = "\n".join([f"{k}: {v}" for k, v in additional_info.items()])
    
    chain_input = {
        "user_context": user_context,
        "user_message": user_message,
        "category": category,
        "subcategory": subcategory,
        "gpa": gpa,
        "extracurriculars": extracurriculars,
        "zipcode": zipcode,
        "high_school_size": high_school_size,
        "additional_info": additional_info_str,
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

def run_counselor():
    print("\n===== Welcome to KYROS AI Counselor =====")
    user_context = input("Please provide some context about yourself (e.g., 'I am a 10th-grade student interested in engineering'): ")
    user_message = input("\nHow can I assist you today?\nYour message: ")
    state = {
        'user_context': user_context,
        'user_message': user_message,
        'user_profile': {},
        'additional_info': {},
    }
    # Step 2: Select category
    state = select_category(state)
    # Step 3: Prerequisite check
    state = prerequisite_check(state)
    if not state['data_check']:
        # Step 4: Collect profile info
        state = collect_profile_info(state)
    # Step 5: Inform user
    state = inform(state)
    # Step 6: Probe for additional details
    state = probe_for_details(state)
    # Step 7: Generate recommendations
    state = recommend(state)
    # Step 8: Get feedback
    state = get_feedback(state)
    # Step 9: Update profile
    state = update_profile(state)
    print("\n===== Thank you for using KYROS AI Counselor! =====\n")

if __name__ == "__main__":
    try:
        run_counselor()
    except Exception as e:
        print(f"An error occurred: {e}")