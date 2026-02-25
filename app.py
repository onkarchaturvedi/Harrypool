import joblib
import random
import gradio as gr

# Load your trained model
artifact = joblib.load("finalmodel.pkl")
pipeline = artifact["pipeline"]
responses = artifact["responses"]

# Define chatbot function
def chat_fn(message, history):
    # Predict intent
    pred = pipeline.predict([message])[0]
    # Pick random response
    reply = random.choice(responses.get(pred, ["Sorry, I don’t know about that."]))
    return reply

# Create a ChatInterface (chat bubble style like ChatGPT)
demo = gr.ChatInterface(
    fn=chat_fn,
    title="Harrypool Chatbot 🤖",
    description="Ask me anything about Onkar (Harry) Chaturvedi — education, projects, skills, career, hobbies, and more!",
    theme="default"
)

if __name__ == "__main__":
    demo.launch()
