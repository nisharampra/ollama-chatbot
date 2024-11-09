from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_ollama import OllamaLLM  
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Define the chat agent class
class ChatAgent:
    def __init__(self, llm):
        self.llm = llm

    def create_chat_prompt(self, user_input):
        return f"""You are a friendly assistant. Respond to the following message with an appropriate answer.
        User: {user_input}
        Assistant:"""

    def chat(self, user_input):
        chat_prompt = self.create_chat_prompt(user_input)
        chat_chain = LLMChain(prompt=PromptTemplate(template=chat_prompt), llm=self.llm)
        response = chat_chain.run({"user_input": user_input})
        return response

# Initialize chatbot
def initialize_chatbot():
    # Initialize LLM (Ollama) with the required model
    llm = OllamaLLM(model="llama3.2", callbacks=[StreamingStdOutCallbackHandler()])
    chat_agent = ChatAgent(llm)
    return chat_agent

# Main loop to handle continuous chatbot interaction
def chatbot():
    chat_agent = initialize_chatbot()
    
    print("Chatbot ready. Type 'exit' to quit.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if user_input.strip() == "":
            continue

        response = chat_agent.chat(user_input)
        print("\nAssistant:", response)

if __name__ == "__main__":
    chatbot()
