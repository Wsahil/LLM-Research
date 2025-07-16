import streamlit as st
import cohere

# Session state to track whether the API key has been entered
if "api_key_entered" not in st.session_state:
    st.session_state.api_key_entered = False

# UI for entering the API key (shown only if not already entered)
if not st.session_state.api_key_entered:
    api_key = st.text_input("Enter your Cohere API Key to initialize", type="password")
    if api_key:
        st.session_state.api_key_entered = True
        st.session_state.api_key = api_key
else:
    api_key = st.session_state.api_key

# If no API key is provided, warn the user and stop the app
if not api_key:
    st.warning("Please enter your API key to continue.")
    st.stop()

# Initialize the Cohere client with the provided API key
cohere_client = cohere.Client(api_key)


# Function to query Cohere and get a response with customized prompts
def query_cohere(prompt, tone="neutral", detail="medium"):
    # Setting up the system prompt based on tone and detail preferences
    system_prompt = f"The user has asked a question. Respond in a {tone} tone with {detail} detail. If you don't know the answer, respond with 'I do not know"
    prompt = system_prompt + prompt
    max_tokens = 300 if detail == "detailed" else 100
    
    response = cohere_client.generate(prompt=prompt, max_tokens=max_tokens)
    bot_response = response.generations[0].text.strip()
    
    return bot_response


# Initialize chat history if not present
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

st.sidebar.title("Response Settings")
st.sidebar.write("Select tone and detail level for responses.")

tone = st.sidebar.radio("Tone", ["Formal", "Casual", "Humorous"], key="tone_selector")
detail = st.sidebar.radio("Detail Level", ["Brief", "Moderate", "Detailed"], key="detail_selector")

st.session_state["tone"] = tone
st.session_state["detail"] = detail


# Display welcome messages if history is empty
if len(st.session_state.conversation_history) == 0:
    st.session_state.conversation_history.append("Chatbot: Welcome to Cohere chatbot!")
    st.session_state.conversation_history.append("Chatbot: How can I assist you today?")
    st.session_state.conversation_history.append("Chatbot: Type 'quit' to end the conversation.")

# Quit button with tooltip to end the chat session
if st.button("Quit", help="End the conversation and clear the history.", key="quit_button"):
    st.session_state.conversation_history = []
    st.session_state.api_key_entered = False
    st.write("Chatbot: Goodbye! The conversation has ended.")
    st.stop()

# User input field
user_input = st.chat_input("You: ")


if user_input:
    if user_input.lower() in ['exit', 'quit']:
        st.session_state.conversation_history.append("Chatbot: Goodbye!")
        st.session_state.api_key_entered = False
        st.stop()

    # Add user input to conversation history
    st.session_state.conversation_history.append(f"You: {user_input}")

    # Query the LLM and generate the bot response with selected tone and detail level
    prompt = "\n".join(st.session_state.conversation_history) + "\nChatbot: "
    bot_response = query_cohere(prompt, tone=tone.lower(), detail=detail.lower())

    # Add bot response to conversation history
    st.session_state.conversation_history.append(f"Chatbot: {bot_response}")

# Display the conversation history incrementally with labels
for message in st.session_state.conversation_history:
    if message.startswith("You:"):
        st.markdown(f"<span style='color:orange; font-weight:bold;'>You:</span> {message[4:]}", unsafe_allow_html=True)
    else:
        st.markdown(f"<span style='color:brown; font-weight:bold;'>Chatbot:</span> {message[9:]}", unsafe_allow_html=True)


