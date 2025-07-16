import streamlit as st
import cohere
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import re

# Initialize Pinecone
PINECONE_API_KEY = "pcsk_2xrWik_8aW44s4muE68taxW7YG2NztkQiTTDjPifHYcdyxDtCDbWaRR1Vq2Q2HrBSVEok5"
PINECONE_ENV = "us-east-1"
INDEX_NAME = "ischool-courses"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Setup Cohere
if "api_key_entered" not in st.session_state:
    st.session_state.api_key_entered = False

if not st.session_state.api_key_entered:
    api_key = st.text_input("Enter your Cohere API Key", type="password")
    if api_key:
        st.session_state.api_key_entered = True
        st.session_state.api_key = api_key
else:
    api_key = st.session_state.api_key

if not api_key:
    st.warning("Please enter your API key to continue.")
    st.stop()

cohere_client = cohere.Client(api_key)


# Function 1: Course description
def get_course_description(course_name):
    results = index.query(vector=model.encode(course_name.strip().lower()).tolist(), top_k=1, include_metadata=True)
    if results and "matches" in results:
        return results["matches"][0]["metadata"]["description"]
    return "Course description not found."

# Function 2: Course credits
def get_course_credits(course_name):
    description = get_course_description(course_name)
    match = re.search(r"(\d+[-–]?\d*) credit\(s\)", description)
    return f"This course offers {match.group(1)} credit(s)." if match else "Credit information not available."

# Function 3: Course schedule
def get_course_schedule(course_name):
    description = get_course_description(course_name)
    if "fall" in description.lower() or "spring" in description.lower():
        return "This course is offered in Fall and/or Spring semesters."
    elif "irregularly" in description.lower():
        return "This course is offered on an irregular basis."
    return "Schedule information not available."



# Dispatcher
def handle_functional_query(user_input):
    if "credit" in user_input.lower() or "credits" in user_input.lower():
        return get_course_credits(user_input)
    elif "schedule" in user_input.lower() or "offered" in user_input.lower():
        return get_course_schedule(user_input)
    elif "description" in user_input.lower() or "about" in user_input.lower():
        return get_course_description(user_input)
    return None


def query_pinecone(query_text, top_k=3):
    query_embedding = model.encode(query_text).tolist()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    
    if results and "matches" in results:
        courses = [
            f"- **{match['metadata']['course_name']}**\n  {match['metadata'].get('description', '')}"
            for match in results['matches']
        ]
        return "\n\n".join(courses) if courses else "No matching courses found."
    return "No matching courses found."


# Cohere fallback
def query_cohere(user_input):
    retrieved_info = query_pinecone(user_input)

    prompt = f"""The user asked: "{user_input}"
    
    Here are some relevant Syracuse University graduate courses related to the query:
    
    {retrieved_info}
    
    Based on this, generate a helpful response for the user.
    """
    response = cohere_client.generate(prompt=prompt, max_tokens=300)
    return response.generations[0].text.strip()


# Streamlit UI
st.title("School of Informaiton Studies Course Assistant Bot")
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = ["Chatbot: Welcome! Ask about graduate courses at iSchool of Syracuse University."]

user_input = st.chat_input("Ask about a course (e.g., credits, schedule, description)...")

if user_input:
    st.session_state.conversation_history.append(f"You: {user_input}")
    answer = handle_functional_query(user_input)
    if not answer:
        answer = query_cohere(user_input)
    st.session_state.conversation_history.append(f"Bot: {answer}")

for message in st.session_state.conversation_history:
    if message.startswith("You:"):
        st.markdown(f"<span style='color:orange; font-weight:bold;'>You:</span> {message[4:]}", unsafe_allow_html=True)
    else:
        st.markdown(f"<span style='color:brown; font-weight:bold;'>Chatbot:</span> {message[9:]}", unsafe_allow_html=True)
