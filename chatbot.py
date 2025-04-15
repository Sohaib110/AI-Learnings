import re

responses = {
    "hello": "Hello! How can I assist you today?",
    "hi": "Hi there! How can I help you?",
    "how are you": "I'm just a computer program, but thanks for asking! How can I assist you?",
    "what is your name": "I am a simple chatbot created for demonstration purposes.",
    "help": "Sure! What do you need help with?",
    "bye": "Goodbye! Have a great day!",
    "thank you": "You're welcome! If you have any more questions, feel free to ask.",
    "default": "I'm sorry, I didn't understand that. Can you please rephrase?"
}

def chatbot_response(user_input):
    user_input = user_input.lower().strip()
    
    # Check for exact matches first
    if user_input in responses:
        return responses[user_input]
    
    # Then check for partial matches
    for keyword in responses:
        if keyword in user_input and keyword != "default":
            return responses[keyword]
    
    return responses["default"]
    
def chatbot():
    print("Chatbot: Hello! I am here to assist you. Type 'bye' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "bye":
            print("Chatbot: Goodbye! Have a great day!")
            break
        
        response = chatbot_response(user_input)
        print(f"Chatbot: {response}")
            
chatbot()