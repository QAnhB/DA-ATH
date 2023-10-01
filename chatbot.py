import streamlit as st
from PIL import Image
import io
from image import ImageClassification
from text import TextExtract

# Title
st.title("Chatbot")

# Create a sidebar for instructions
st.sidebar.markdown("### Instructions")
st.sidebar.markdown("1. Type your message in the input box.")
st.sidebar.markdown("2. Attach a file if needed.")
st.sidebar.markdown("3. Click 'Send' to chat with the bot.")

# Chatbot conversation history
conversation = []

# Chatbot response function (You can replace this with your NLP model)
def chatbot_response(user_input):
    # Replace this with your NLP model logic
    return f"Chatbot: You said '{user_input}'"

# User input text box
user_input = st.text_area("You:", value="", key="user_input")

# File upload
uploaded_file = st.file_uploader("Attach File:", type=['jpg','png','jpeg'])

# Send button
if st.button("Send"):
    if user_input:
        conversation.append(f"You: {user_input}")
        response = TextExtract(user_input)
        text = response.keywords_detect()
        conversation.append(f"Chatbot: {text}")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        model = ImageClassification(image)
        text = model.describe()
        conversation.append(f"Chatbot: {text}")
        st.image(image,use_column_width=True)


# Display conversation
st.markdown("<hr>", unsafe_allow_html=True)
for message in conversation:
    st.write(message)

# Clear conversation
if st.button("Clear Conversation"):
    conversation = []
