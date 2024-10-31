# Databricks notebook source
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
# from langchain.chat_models import ChatDatabricks
from langchain_databricks.chat_models import ChatDatabricks
from langchain.vectorstores import DatabricksVectorSearch
from databricks.vector_search.client import VectorSearchClient

from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

llm_dbrx = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens=200, disable_streaming=False)
vs_endpoint_name = "cubie_vs_endpoint"
vs_index_table_fullname = "workspace.cubie.customer_embeddings"
prompt_template_cubie = PromptTemplate.from_template(
    """
    You are Cubie, a friendly chatbot designed to assist customers with self-storage reservations. You have to greet customer in a friendly manner. Make sure you introduce your name as Cubie. Sometimes, the customer needs help on something else. If the customer needs help about something other than reservation or choosing appropriate unit for their need, you have to say that you will contact a customer service for them. Your job is to help them find a unit and make reservation only. Your goal is to provide accurate information and help users make reservations based on their needs. Only recommend the units that are available within the location with the zipcode of the customer's request. Do not recommend a unit without knowing where the customer want to rent. You have to ask where the customer want to rent first before recommending! Only show available units only! When recommend the units, do not show the ID, but only show the facility name, facility address or unit name or size. Make sure that the unit is available.

    Here are some relevant details you can use to answer customer inquiries:

    <context>
    {context}
    </context>

    Customer Question: {input}

    Your Response:
    """
)

# Function to get the retriever from the Databricks vector store
def get_retriever():
    vsc = VectorSearchClient(disable_notice=True)
    vs_index = vsc.get_index(vs_endpoint_name, vs_index_table_fullname)
    # Create a vector store from the index, using 'text' as the text column
    vectorstore = DatabricksVectorSearch(vs_index, text_column="text")
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# Construct the chain for question-answering
question_answer_chain = create_stuff_documents_chain(llm_dbrx, prompt_template_cubie)
chain1 = create_retrieval_chain(get_retriever(), question_answer_chain)

def cubie_chatbot_tool(input_text: str) -> str:
    try:
        # Invoke the chain with the customer's input and retrieve the response
        response = chain1.invoke({"input": input_text})
        
        # Debugging: Print the response to inspect its structure
        print("Response received from chain1.invoke:", response)
        
        # Check if 'answer' is the correct key
        if 'answer' in response:
            return response['answer']
        else:
            # Adjust this based on the actual response structure
            return response  # Or adjust to a different key if needed
    except Exception as e:
        print("Error invoking the chain:", e)
        return "I'm sorry, I encountered an error while processing your request."


# Wrap the tool function in a LangChain Tool
cubie_tool = Tool(
    name="CubieChatbot",
    func=cubie_chatbot_tool,
    description="Use this tool to help customers with self-storage reservations and unit recommendations."
)

# Initialize the agent with the Cubie tool
agent = initialize_agent(
    tools=[cubie_tool],
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    llm=llm_dbrx,
    verbose=True
)

# Example usage
query = "Hello, I want to make a reservation"
response = agent.run(input=query)
print(response) 


# COMMAND ----------

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
import pandas as pd

# Define input and output schema
input_schema = Schema([
    ColSpec("string", "input") 
])
output_schema = Schema([
    ColSpec("string", "answer") 
])

# Create a model signature
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define a custom MLflow model wrapper
class CubieChatbotWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, chain):
        self.chain = chain
    
    def predict(self, context, model_input):
        input_text = model_input["input"].iloc[0]  
        response = self.chain.invoke({"input": input_text})
        return pd.Series([response['answer']])  

# Save and log the model with signature
with mlflow.start_run() as run:
    cubie_model = CubieChatbotWrapper(chain=chain1)
    mlflow.pyfunc.log_model(
        "cubie_chatbot_model", 
        python_model=cubie_model, 
        signature=signature, 
        conda_env="conda.yml"
    )
    model_uri = f"runs:/{run.info.run_id}/cubie_chatbot_model"
    model_version = mlflow.register_model(model_uri, "CubieChatbotModel")

