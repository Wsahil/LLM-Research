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

# defining the function to query Cohere and get a response
def query_cohere(prompt):
    response = cohere_client.generate(
        prompt=prompt,
        max_tokens=100  # Limit the response length
    )
    return response.generations[0].text.strip()

# initialize chat history if not present
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Welcome messages
if len(st.session_state.conversation_history) == 0:
    st.session_state.conversation_history.append("Cohere:  Welcome to Cohere chatbot!")
    st.session_state.conversation_history.append("Cohere:  Hello! How can I assist you today?")
    st.session_state.conversation_history.append("Cohere:  Type 'quit' to end the conversation.")

# Quit button with tooltip to end the chat session
if st.button("Quit", help="This will delete the conversation history and you will need to reenter your API key to start again.", key="quit_button"):
    st.session_state.conversation_history = []  # Clear conversation history
    st.session_state.api_key_entered = False  # Reset API key
    st.write("Chatbot: Goodbye! The conversation has ended.")
    st.stop()


user_input = st.chat_input("You: ")

if user_input:
    if user_input.lower() in ['exit', 'quit']:
        st.session_state.conversation_history.append("Chatbot: Goodbye!")
        st.session_state.api_key_entered = False
        st.stop()

    # Add user input to conversation history
    st.session_state.conversation_history.append(f"You: {user_input}")

    # Query the LLM and generate the bot response
    prompt = "\n".join(st.session_state.conversation_history) + "\nChatbot: "
    bot_response = query_cohere(prompt)

    # Add bot response to conversation history
    st.session_state.conversation_history.append(f"Chatbot: {bot_response}")

# Display the conversation history incrementally
# for message in st.session_state.conversation_history:
#         st.write(message)

# Display the conversation history incrementally with highlighted labels
for message in st.session_state.conversation_history:
    if message.startswith("You:"):
        # Highlight 'You:' and keep the rest of the message normal
        st.markdown(f"<span style='color:orange; font-weight:bold;'>You:</span> {message[4:]}", unsafe_allow_html=True)
    else:
        # Highlight 'Chatbot:' and keep the rest of the message normal
        st.markdown(f"<span style='color:brown; font-weight:bold;'>Chatbot:</span> {message[9:]}", unsafe_allow_html=True)


