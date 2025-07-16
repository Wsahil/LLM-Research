import streamlit as st
import cohere
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone



PINECONE_API_KEY = "pcsk_2xrWik_8aW44s4muE68taxW7YG2NztkQiTTDjPifHYcdyxDtCDbWaRR1Vq2Q2HrBSVEok5"
PINECONE_ENV = "us-east-1"
INDEX_NAME = "ischool-courses"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)


model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Cohere
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


# Function to query Pinecone vector database
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


# Function to generate response from Cohere
def query_cohere(user_input):
    retrieved_info = query_pinecone(user_input)

    prompt = f"""The user asked: "{user_input}"
    
    Here are some relevant Syracuse University graduate courses related to the query:
    
    {retrieved_info}
    
    Based on this, generate a helpful response for the user.
    """

    response = cohere_client.generate(prompt=prompt, max_tokens=300)
    return response.generations[0].text.strip()


# Chatbot UI with Streamlit
st.sidebar.title("Chatbot Settings")
st.sidebar.write("Customize response settings.")
tone = st.sidebar.radio("Tone", ["Formal", "Casual", "Professional"], key="tone_selector")
detail = st.sidebar.radio("Detail Level", ["Brief", "Moderate", "Detailed"], key="detail_selector")

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = ["Chatbot: Welcome! Ask about graduate courses at Syracuse University."]

user_input = st.chat_input("Ask about a course...")

if user_input:
    st.session_state.conversation_history.append(f"You: {user_input}")
    
    response = query_cohere(user_input)
    st.session_state.conversation_history.append(f"Chatbot: {response}")

for message in st.session_state.conversation_history:
    if message.startswith("You:"):
        st.markdown(f"<span style='color:orange; font-weight:bold;'>You:</span> {message[4:]}", unsafe_allow_html=True)
    else:
        st.markdown(f"<span style='color:brown; font-weight:bold;'>Chatbot:</span> {message[9:]}", unsafe_allow_html=True)
