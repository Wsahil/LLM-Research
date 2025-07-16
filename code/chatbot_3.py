import cohere

# Initialize the Cohere client with your API key
cohere_client = cohere.Client('Twbor4crkf5gl2i8kaKKUBYxd5cwafcodpOIeV6H')

# Function to query Cohere and get a response
def query_cohere(prompt):
    response = cohere_client.generate(
        prompt=prompt,
        max_tokens=100  # Limit the response length
    )
    return response.generations[0].text.strip()

# Start the conversation loop
conversation_history = ""
print("Cohere: Welcome to Cohere chatbot!")
print("Cohere: Hello! How can I assist you today?")
print("Cohere: Type 'quit' to exit the conversation.")

# Keep the conversation going until the user types 'quit' or 'exit'
while True:
    user_input = input("You: ")

    # Exit the chat if the user requests it
    if user_input.lower() in ['exit', 'quit']:
        print("Chatbot: Goodbye!")
        break
    
    # Update the conversation history with the user's input
    conversation_history += f"You: {user_input}\n"
    
    # Send the conversation history to Cohere and get the bot's response
    prompt = f"{conversation_history}Chatbot: "
    bot_response = query_cohere(prompt)
    
    # Add the bot's response to the conversation history
    conversation_history += f"Chatbot: {bot_response}\n"
    
    # Display the bot's response
    print(f"Chatbot: {bot_response}")
