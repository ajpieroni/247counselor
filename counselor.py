# counselor.py

# Standard Libraries
import os
import logging
import json
import requests  # For web search (simulated in this script)
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
    "Enrichment Opportunities": ["Summer Programs", "Internships", "Workshops"],
    "Personal Development": ["Self-Reflection", "Growth Comparisons"],
    "College Applications": ["Essay Guidance", "Application Tracker", "College List", "Scholarships"],
}

# Define the GraphState
class GraphState(TypedDict):
    user_name: str
    user_grade_level: int
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
    action_items: List[Dict[str, str]]  # To store action items
    web_results: List[str]  # To store web search results
    data_to_store: str  # To store data that should be saved

# Counselor response generation prompt
generate_prompt = PromptTemplate(
    template="""
You are an experienced and empathetic college counselor named Kyros, providing personalized advice to high school students. Engage with the student in a friendly and supportive manner, addressing them by their name. Offer tailored guidance based on the student's profile, additional information, and web search results. Be as specific and personalized as possible, addressing the student's individual situation, goals, and challenges. Incorporate the web search findings into your recommendations to make them hyper-specific. Ensure that any recommended programs accept students of {user_grade_level} grade level and mention any prerequisites. Include links to programs or resources when appropriate. Focus on the category of {category} and subcategory of {subcategory}.

Generate a list of action items for the student that are SMART (Specific, Measurable, Achievable, Relevant, Time-bound), and time-bound by season (e.g., Fall, Winter, Spring, Summer). These action items should help the student achieve their goals.

At the end, if there is any important data from the conversation that should be stored in the student's profile for future reference, note it explicitly under "Data to Store".

Student's Name: {user_name}
Student's Context: {user_context}

Student's Profile:
GPA: {gpa}
Extracurricular Activities: {extracurriculars}
Zipcode: {zipcode}
High School Size: {high_school_size}

Additional Information:
{additional_info}

Web Search Results:
{web_results}

Student's Message: {user_message}
""",
    input_variables=[
        "user_name",
        "user_grade_level",
        "user_context",
        "user_message",
        "category",
        "subcategory",
        "gpa",
        "extracurriculars",
        "zipcode",
        "high_school_size",
        "additional_info",
        "web_results",
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

# Question generation prompt
question_generation_prompt = PromptTemplate(
    template="""
You are an experienced college counselor. Based on the student's context and message, generate a list of leading questions to ask the student to gather more information. The questions should be open-ended and encourage the student to share more about their interests, motivations, and goals.

Student's Context: {user_context}
Student's Message: {user_message}

Generate up to 5 relevant questions.
""",
    input_variables=["user_context", "user_message"],
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
            print(f"\nI've identified your category as {category}.")
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
                            print(f"Great choice! You've selected subcategory: {subcategory}")
                            break
                        else:
                            print("Invalid selection. Please try again.")
                    except ValueError:
                        print("Please enter a valid number.")
            else:
                print("No subcategories available for the selected category.")
        else:
            print(f"\nI've identified your category as {category} and subcategory as {subcategory}.")
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
    response = f"\nThank you! Here's the information we have so far:\n{known_data}"
    print(response)
    state['response'] = response
    return state

def probe_for_details(state):
    print("\n===== Step: Gathering Additional Information =====")
    user_context = state['user_context']
    user_message = state['user_message']
    # Generate leading questions using LLM
    chain_input = {
        "user_context": user_context,
        "user_message": user_message,
    }
    prompt = question_generation_prompt.format(**chain_input)
    messages = [HumanMessage(content=prompt)]
    response = llama2(messages)
    questions_text = response.content.strip()
    # Split the questions into a list
    questions = [q.strip('- ').strip() for q in questions_text.split('\n') if q.strip()]
    additional_info = {}
    for question in questions:
        answer = input(f"{question}\nYour answer: ")
        additional_info[question] = answer
    state['additional_info'] = additional_info
    return state

def perform_web_search(state):
    print("\n===== Step: Performing Web Search =====")
    # Prepare search query based on user information
    user_profile = state.get('user_profile', {})
    additional_info = state.get('additional_info', {})
    grade_level = state.get('user_grade_level', 11)
    search_query = f"{state['user_context']} {state['user_message']}"
    search_query += " " + " ".join(additional_info.values())

    # Simulated web search results tailored to the student's grade level and interests
    web_results = []

    # Simulate checking for programs that accept the student's grade level
    if 'archaeology' in state['user_context'].lower() or 'archaeology' in state['user_message'].lower():
        # Add archaeology programs
        web_results.append("Archaeology Summer Program at University of California, Los Angeles (UCLA) accepting applications from 12th graders")
        web_results.append("High School Archaeology Program at Boston University for graduating seniors")
        web_results.append("Local volunteer opportunities in archaeological digs near zipcode " + user_profile.get('zipcode', ''))
    else:
        # General programs
        web_results.append("General enrichment programs for high school students")

    state['web_results'] = web_results

    print("Web search completed. Results obtained.")
    return state

def recommend(state):
    print("\n===== Step: Generating Personalized Recommendations =====")
    user_name = state.get('user_name', '')
    user_grade_level = state.get('user_grade_level', 11)
    user_context = state['user_context']
    user_message = state['user_message']
    category = state['category']
    subcategory = state['subcategory']
    user_profile = state.get('user_profile', {})
    additional_info = state.get('additional_info', {})
    web_results = state.get('web_results', [])
    gpa = user_profile.get('gpa', 'N/A')
    extracurriculars = user_profile.get('extracurriculars', 'N/A')
    zipcode = user_profile.get('zipcode', 'N/A')
    high_school_size = user_profile.get('high_school_size', 'N/A')
    additional_info_str = "\n".join([f"{k}: {v}" for k, v in additional_info.items()])
    web_results_str = "\n".join(web_results)

    chain_input = {
        "user_name": user_name,
        "user_grade_level": user_grade_level,
        "user_context": user_context,
        "user_message": user_message,
        "category": category,
        "subcategory": subcategory,
        "gpa": gpa,
        "extracurriculars": extracurriculars,
        "zipcode": zipcode,
        "high_school_size": high_school_size,
        "additional_info": additional_info_str,
        "web_results": web_results_str,
    }
    prompt = generate_prompt.format(**chain_input)
    messages = [HumanMessage(content=prompt)]
    response = llama2(messages)
    output_text = response.content.strip()

    # Extract recommendations and action items
    if "Action Items:" in output_text:
        recommendations_text, action_items_section = output_text.split("Action Items:", 1)
    else:
        recommendations_text = output_text
        action_items_section = ""

    # Extract data to store if any
    if "Data to Store:" in action_items_section:
        action_items_text, data_to_store_text = action_items_section.split("Data to Store:", 1)
        data_to_store = data_to_store_text.strip()
    else:
        action_items_text = action_items_section
        data_to_store = ""

    # Parse action items
    action_items = []
    if action_items_text:
        lines = action_items_text.strip().split('\n')
        for line in lines:
            if line.strip() and (line[0].isdigit() or line.strip().startswith('-')):
                action_item = line.strip().lstrip('- ').lstrip('1234567890. ').strip()
                action_items.append(action_item)
    state['action_items'] = action_items

    # Explicitly call out data to store
    if data_to_store:
        print("\n===== Data to Store =====")
        print(data_to_store)
        state['data_to_store'] = data_to_store

    recommendations = [recommendations_text.strip()]
    print("\nRecommendations:")
    print(f"{recommendations_text.strip()}")
    state['recommendations'] = recommendations
    return state

def action_items_selection(state):
    if state.get('action_items'):
        print("\n===== Action Items =====")
        print("Here are some action items for you:")
        for idx, item in enumerate(state['action_items'], 1):
            print(f"{idx}. {item}")
        # Prompt user to select action items to add to roadmap planning
        selected_items = []
        while True:
            selection = input("\nWould you like to add any of these to your roadmap planning? Type the number(s), separated by commas, or 'no' to skip: ")
            if selection.lower() == 'no':
                break
            try:
                indices = [int(x.strip()) for x in selection.split(',') if x.strip()]
                for idx in indices:
                    if 1 <= idx <= len(state['action_items']):
                        selected_items.append(state['action_items'][idx - 1])
                    else:
                        print(f"Invalid selection: {idx}")
                break
            except ValueError:
                print("Please enter valid numbers separated by commas, or 'no' to skip.")
        if selected_items:
            # Print the action items that will be sent via JSON to roadmap planning
            print("\n===== Action Items to Add to Roadmap Planning =====")
            for item in selected_items:
                print(f"- {item}")
            # Simulate sending to roadmap planning
            action_items_json = json.dumps({"action_items": selected_items}, indent=2)
            print("\nAction items to be sent to roadmap planning (in JSON format):")
            print(action_items_json)
            state['selected_action_items'] = selected_items
        else:
            print("No action items were selected to add to roadmap planning.")
    else:
        print("\nNo action items were generated.")
    return state

def get_feedback(state):
    print("\n===== Step: Collecting Feedback =====")
    feedback = input("Was this information helpful? (Yes/No): ")
    if feedback.strip().lower() == 'yes':
        print("I'm glad I could help!")
    else:
        print("I'm sorry to hear that. I'll strive to provide better assistance next time.")
    state['feedback'] = feedback
    return state

def update_profile(state):
    print("\n===== Step: Updating Your Profile =====")
    # Update the user_profile with data_to_store
    if 'data_to_store' in state:
        print("Storing the following data to your profile:")
        print(state['data_to_store'])
        # For simplicity, let's assume data_to_store is in key: value format
        for line in state['data_to_store'].split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                state['user_profile'][key.strip()] = value.strip()
    else:
        print("No new data to store in your profile.")
    print("Your profile has been updated with the new information.")
    return state

def run_counselor():
    print("\n===== Welcome to Kyros AI College Counselor =====")
    user_name = input("To get started, may I have your name? ")
    print(f"Nice to meet you, {user_name}!")
    # Ask for grade level
    while True:
        grade_input = input("Please enter your current grade level (9, 10, 11, or 12): ")
        try:
            user_grade_level = int(grade_input)
            if user_grade_level in [9, 10, 11, 12]:
                break
            else:
                print("Please enter a valid grade level (9, 10, 11, or 12).")
        except ValueError:
            print("Please enter a number (9, 10, 11, or 12).")
    user_context = input("Please provide some context about yourself (e.g., 'I'm interested in engineering'): ")
    user_message = input("\nHow can I assist you today?\nYour message: ")
    state = {
        'user_name': user_name,
        'user_grade_level': user_grade_level,
        'user_context': user_context,
        'user_message': user_message,
        'user_profile': {},
        'additional_info': {},
    }
    # Proceed through the steps
    state = select_category(state)
    state = prerequisite_check(state)
    if not state['data_check']:
        state = collect_profile_info(state)
    state = inform(state)
    state = probe_for_details(state)
    state = perform_web_search(state)
    state = recommend(state)
    state = action_items_selection(state)
    state = get_feedback(state)
    state = update_profile(state)
    print(f"\n===== Thank you for using Kyros AI College Counselor, {user_name}! Good luck with your endeavors! =====\n")

if __name__ == "__main__":
    try:
        run_counselor()
    except Exception as e:
        print(f"An error occurred: {e}")