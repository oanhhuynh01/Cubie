import logging
import os
import streamlit as st
import requests
import json

# Set up logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["SERVING_ENDPOINT"] = "https://dbc-ab436d26-360e.cloud.databricks.com/serving-endpoints/cubiechat10/invocations"
# Assuming this is the function you use to call your chatbot model
def cubie_chatbot_tool(input_text):
    endpoint_url = os.getenv("SERVING_ENDPOINT")
    logger.info("Using endpoint URL: %s", endpoint_url)

    # Prepare the input data according to expected schema
    # data = {
    #     "input": input_text  # Changed to match model schema
    # }
    data = {
    "inputs": [input_text]  # Adjusted based on MLflow model serving expectations
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer dapid109598037cfc5641cd22fe3e7f6e2df"  
    }

    try:
        # Make the POST request to the serving endpoint
        response = requests.post(endpoint_url, headers=headers, data=json.dumps(data))

        # Log the response status code and content
        logger.info("Response Status Code: %s", response.status_code)
        logger.info("Response Content: %s", response.text)

        # Check if the request was successful
        response.raise_for_status()

        # Parse the response
        result = response.json()
        return result.get('answer', 'No answer found.')  # Adjust based on response structure

    except requests.exceptions.RequestException as e:
        logger.error("Error calling model: %s", e)
        return "Sorry, there was an error processing your request."

# Streamlit app
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False


# UI configurations
st.set_page_config(page_title="Cubie",
                   page_icon=":large_red_square:",
                   layout="wide")

col1, col2 = st.columns([1, 4])

with col1:
    st.image("Cubie.jpg", width=150)

with col2:
    st.title("Hello! I'm Cubie")
    st.write("Your go-to AI assistant for self-storage needs")

# Sidebar content
st.sidebar.image("storage_unit.png", width=100)
st.sidebar.title("Cubie by SmartCube")
st.sidebar.write("AI-powered virtual assistant for all things self-storage.")

# Display services in the sidebar
st.sidebar.subheader("Cubie can help you with:")
services = [
    ("🔍", "Checking Availability"),
    ("🆕", "Making a New Reservation"),
    ("📦", "Finding the Perfect Storage Unit size"),
    ("📅", "Account Status"),
    ("📞", "Create a Support Case"),
    ("🤝", "Connecting You with the Right Team for Call Assistance"),
]
for icon, service in services:
    st.sidebar.write(f"{icon} {service}")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call the chatbot model
    assistant_response = "..."  # Initialize the response to avoid NameError
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        try:
            logger.info("Querying model with input: %s", prompt)
            assistant_response = cubie_chatbot_tool(prompt)  # Call the function
            logger.info("Received response: %s", assistant_response)
            st.markdown(assistant_response)
        except Exception as e:
            logger.error(f"Error querying model: {e}")
            st.error(f"Error querying model: {e}")

    # Only add to chat history if the response is defined
    if assistant_response != "...":
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

# -------------------------
