import streamlit as st
import speech_recognition as sr
from transformers import pipeline

# Load chatbot model
@st.cache_resource
def load_chatbot():
    return pipeline("text2text-generation", model="facebook/blenderbot-400M-distill")

chatbot = load_chatbot()

def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        try:
            audio = recognizer.listen(source, timeout=5)
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand."
        except sr.RequestError:
            return "Speech service error."
        except sr.WaitTimeoutError:
            return "No speech detected."

# Streamlit UI
st.title("🤖 AI Chatbot with Voice & Text Input")
st.write("Talk to me or type your message!")

# Maintain chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input options
col1, col2 = st.columns([2, 1])
with col1:
    user_input = st.text_area("Type your message:", "", height=100)
with col2:
    if st.button("🎤 Speak"):
        user_input = speech_to_text()
        st.write(f"You said: {user_input}")

if st.button("Send") and user_input.strip():
    with st.spinner("Thinking..."):
        bot_response = chatbot(user_input, max_length=100, do_sample=True, temperature=0.7)
        response_text = bot_response[0]['generated_text']
        
        # Store conversation history
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response_text))

# Display chat history
st.subheader("Chat History")
for sender, message in st.session_state.chat_history:
    with st.chat_message("user" if sender == "You" else "assistant"):
        st.write(f"**{sender}:** {message}")
