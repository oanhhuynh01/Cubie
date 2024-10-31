# Databricks notebook source
# MAGIC %pip install -U --quiet databricks-sdk==0.29.0 langchain-core==0.2.24 databricks-vectorsearch==0.40 langchain-community==0.2.10 typing-extensions==4.12.2 youtube_search Wikipedia grandalf mlflow==2.14.3 pydantic==2.8.2
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd

# COMMAND ----------

df_facility_unit = spark.sql("Select * from cubie.tbl_facility_unit")

# COMMAND ----------

df_rentals = spark.sql("Select * from cubie.tbl_rentals")

# COMMAND ----------

# Import necessary libraries
from databricks.vector_search.client import VectorSearchClient
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat, concat_ws, lit, coalesce, when

# Initialize Spark session if not already done
spark = SparkSession.builder.getOrCreate()

# Load facility unit and rentals data
df_facility_unit = spark.sql("SELECT * FROM cubie.tbl_facility_unit")
df_rentals = spark.sql("SELECT * FROM cubie.tbl_rentals")

# Join the two tables based on FacilityID and UnitID
df_combined = df_facility_unit.join(df_rentals, on=['FacilityID', 'UnitID'], how='left')

# Create a 'text' column combining relevant data for vector search
df_combined = df_combined.withColumn("text", concat_ws(", ", 
                                                       concat(lit("Facility ID: "), col("FacilityID")),
                                                       concat(lit("Facility Name: "), col("FacilityName")), 
                                                       concat(lit("Facility Address: "), col("FacilityAddress")), 
                                                       concat(lit("Facility ZipCode: "), col("Zipcode")),
                                                       concat(lit("Facility Phone: "), col("FacilityPhone")),
                                                       concat(lit("Facility/Store hours: "), col("Store_hours")),
                                                       concat(lit("Unit ID: "), col("UnitID")),
                                                       concat(lit("Unit Size: "), col("SquareFeet"), lit(" sqft")), 
                                                       concat(lit("Unit Dimension: "), col("Dimensions")),
                                                       concat(lit("Climate Control: "), col("ClimateControlled_flag")),
                                                       concat(lit("Unit Floor: "), col("floor_attribute")),
                                                       concat(lit("Unit Rent Rate: $"), col("RentRate")),
                                                       concat(lit("Unit Discount Availability: "), col("Discount_avail")),
                                                       concat(lit("Unit Discount: "), col("Discount_percentage")),
                                                       concat(lit("Unit Availability: "), 
                                                              when(col("Active") == 1, "Unavailable").otherwise("Available"))))

# Select only the relevant columns for vector search
df_combined = df_combined.select("FacilityID", "UnitID", "text")

# Save the combined data as a Delta table for vector search
df_combined.write.format("delta").mode("overwrite").saveAsTable("cubie.product_text")

# Create an empty Delta table for the embeddings
# spark.sql("CREATE TABLE IF NOT EXISTS cubie.product_embeddings (id STRING, embedding ARRAY<FLOAT>)")

# Define vector search parameters
vs_endpoint_name = "cubie_vs_endpoint"
vs_index_table_fullname = "workspace.cubie.product_embeddings"
vs_source_table_fullname = "workspace.cubie.product_text"

# Create vector search client
vsc = VectorSearchClient(disable_notice=True)

# Function to check if an index exists
def index_exists(vsc, endpoint_name, index_name):
    try:
        indexes = vsc.list_indexes(endpoint_name)
        return index_name in indexes
    except Exception as e:
        print(f"Failed to check if index exists: {e}")
        return False

# Function to delete an existing index
def delete_existing_index(vsc, endpoint_name, index_name):
    try:
        vsc.delete_index(endpoint_name, index_name)
        print(f"Deleted existing index: {index_name}")
    except Exception as e:
        print(f"Failed to delete existing index: {e}")

# Check for existing index and delete it if found
if index_exists(vsc, vs_endpoint_name, vs_index_table_fullname):
    print(f"Index {vs_index_table_fullname} already exists. Deleting it...")
    delete_existing_index(vsc, vs_endpoint_name, vs_index_table_fullname)

# Create the index
try:
    print(f"Creating index {vs_index_table_fullname} on endpoint {vs_endpoint_name}...")
    vsc.create_delta_sync_index(
        endpoint_name=vs_endpoint_name,
        index_name=vs_index_table_fullname,
        source_table_name=vs_source_table_fullname,
        pipeline_type="TRIGGERED",
        primary_key="UnitID",  # Adjust primary key based on your table
        embedding_source_column="text",  # Ensure this matches your actual column for embeddings
        embedding_model_endpoint_name="databricks-bge-large-en"  # Specify your embedding model
    )
    print(f"Index {vs_index_table_fullname} created successfully.")
except Exception as e:
    print(f"Failed to create index: {e}")

# Function to wait for the index to be ready
def wait_for_index_to_be_ready(vsc, endpoint_name, index_name):
    print(f"Waiting for index {index_name} to be ready...")
    # Implement a loop here to check the index status if necessary

# Wait for the index to be ready
wait_for_index_to_be_ready(vsc, vs_endpoint_name, vs_index_table_fullname)


# COMMAND ----------

# Import necessary libraries
from databricks.vector_search.client import VectorSearchClient
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat, concat_ws, lit, coalesce, when

# Initialize Spark session if not already done
spark = SparkSession.builder.getOrCreate()

# Load rentals data
df_rentals = spark.sql("SELECT * FROM cubie.tbl_rentals")

# Create a 'text' column combining relevant data for vector search
df_combined = df_rentals.withColumn("text", concat_ws(", ", 
                                                       concat(lit("CustomerID: "), col("CustomerID")), 
                                                       concat(lit("Customer First Name: "), col("CustomerFirstName")),
                                                       concat(lit("Customer Last Name: "), col("CustomerLastName")),
                                                       concat(lit("Customer Full Name: "), col("CustomerFullName")),
                                                       concat(lit("Customer Email: "), col("CustomerEmailAddress")), 
                                                       concat(lit("Customer Phone: "), col("CustomerPhone")),
                                                       concat(lit("Customer Address: "), col("CustomerAddress")),
                                                       concat(lit("Customer ZipCode: "), col("Customerzipcode")),
                                                       concat(lit("Facility ID of the rental: "), col("FacilityID")),
                                                       concat(lit("Unit ID of the rental: "), col("UnitID")), 
                                                       concat(lit("Customer Rental Rent Rate: "), col("CustomerRentRate")), 
                                                       concat(lit("Customer Rent Status: "), col("RentStatus")),
                                                       concat(lit("Rent Start Date: "), col("RentStartDate")),
                                                       concat(lit("Rent End Date: "), col("RentEndate")),
                                                       concat(lit("Reservation Date: "), col("ReservationDate")),
                                                       concat(lit("Desired Move in Date: "), col("Desired_Move_in_Date")),
                                                       concat(lit("Rental Active Flag: "), col("Active"))

                                                       
))


# Select only the relevant columns for vector search
df_combined = df_combined.select("CustomerID", "CustomerFirstName", "text")

# Save the combined data as a Delta table for vector search
df_combined.write.format("delta").mode("overwrite").saveAsTable("cubie.customer_text")

# Create an empty Delta table for the embeddings
# spark.sql("CREATE TABLE IF NOT EXISTS cubie.product_embeddings (id STRING, embedding ARRAY<FLOAT>)")

# Define vector search parameters
vs_endpoint_name = "cubie_vs_endpoint"
vs_index_tablecustomer_fullname = "workspace.cubie.customer_embeddings"
vs_source_tablecustomer_fullname = "workspace.cubie.customer_text"

# Create vector search client
vsc = VectorSearchClient(disable_notice=True)

# Function to check if an index exists
def index_exists(vsc, endpoint_name, index_name):
    try:
        indexes = vsc.list_indexes(endpoint_name)
        return index_name in indexes
    except Exception as e:
        print(f"Failed to check if index exists: {e}")
        return False

# Function to delete an existing index
def delete_existing_index(vsc, endpoint_name, index_name):
    try:
        vsc.delete_index(endpoint_name, index_name)
        print(f"Deleted existing index: {index_name}")
    except Exception as e:
        print(f"Failed to delete existing index: {e}")

# Check for existing index and delete it if found
if index_exists(vsc, vs_endpoint_name, vs_index_tablecustomer_fullname):
    print(f"Index {vs_index_table_fullname} already exists. Deleting it...")
    delete_existing_index(vsc, vs_endpoint_name, vs_index_tablecustomer_fullname)

# Create the index
try:
    print(f"Creating index {vs_index_tablecustomer_fullname} on endpoint {vs_endpoint_name}...")
    vsc.create_delta_sync_index(
        endpoint_name=vs_endpoint_name,
        index_name=vs_index_tablecustomer_fullname,
        source_table_name=vs_source_tablecustomer_fullname,
        pipeline_type="TRIGGERED",
        primary_key="CustomerID",  # Adjust primary key based on your table
        embedding_source_column="text",  # Ensure this matches your actual column for embeddings
        embedding_model_endpoint_name="databricks-bge-large-en"  # Specify your embedding model
    )
    print(f"Index {vs_index_tablecustomer_fullname} created successfully.")
except Exception as e:
    print(f"Failed to create index: {e}")

# Function to wait for the index to be ready
def wait_for_index_to_be_ready(vsc, endpoint_name, index_name):
    print(f"Waiting for index {index_name} to be ready...")
    # Implement a loop here to check the index status if necessary

# Wait for the index to be ready
wait_for_index_to_be_ready(vsc, vs_endpoint_name, vs_index_tablecustomer_fullname)


# COMMAND ----------

# Import necessary libraries
from databricks.vector_search.client import VectorSearchClient
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat, concat_ws, lit, coalesce, when

# Initialize Spark session if not already done
spark = SparkSession.builder.getOrCreate()

# Load facility unit and rentals data
df_transcript = spark.sql("SELECT * FROM cubie.whisper_api_110_call_recordings")

# Create an empty Delta table for the embeddings
# spark.sql("CREATE TABLE IF NOT EXISTS cubie.product_embeddings (id STRING, embedding ARRAY<FLOAT>)")

# Define vector search parameters
vs_endpoint_name = "cubie_vs_endpoint"
vs_index_tabletranscript_fullname = "workspace.cubie.transcript_embeddings"
vs_source_tabletranscript_fullname = "workspace.cubie.whisper_api_110_call_recordings"

# Create vector search client
vsc = VectorSearchClient(disable_notice=True)

# Function to check if an index exists
def index_exists(vsc, endpoint_name, index_name):
    try:
        indexes = vsc.list_indexes(endpoint_name)
        return index_name in indexes
    except Exception as e:
        print(f"Failed to check if index exists: {e}")
        return False

# Function to delete an existing index
def delete_existing_index(vsc, endpoint_name, index_name):
    try:
        vsc.delete_index(endpoint_name, index_name)
        print(f"Deleted existing index: {index_name}")
    except Exception as e:
        print(f"Failed to delete existing index: {e}")

# Check for existing index and delete it if found
if index_exists(vsc, vs_endpoint_name, vs_index_tabletranscript_fullname):
    print(f"Index {vs_index_tabletranscript_fullname} already exists. Deleting it...")
    delete_existing_index(vsc, vs_endpoint_name, vs_index_table_fullname)

# Create the index
try:
    print(f"Creating index {vs_index_tabletranscript_fullname} on endpoint {vs_endpoint_name}...")
    vsc.create_delta_sync_index(
        endpoint_name=vs_endpoint_name,
        index_name=vs_index_tabletranscript_fullname,
        source_table_name=vs_source_tabletranscript_fullname,
        pipeline_type="TRIGGERED",
        primary_key="externalId",  # Adjust primary key based on your table
        embedding_source_column="transcript",  # Ensure this matches your actual column for embeddings
        embedding_model_endpoint_name="databricks-bge-large-en"  # Specify your embedding model
    )
    print(f"Index {vs_index_table_fullname} created successfully.")
except Exception as e:
    print(f"Failed to create index: {e}")

# Function to wait for the index to be ready
def wait_for_index_to_be_ready(vsc, endpoint_name, index_name):
    print(f"Waiting for index {index_name} to be ready...")
    # Implement a loop here to check the index status if necessary

# Wait for the index to be ready
wait_for_index_to_be_ready(vsc, vs_endpoint_name, vs_index_tabletranscript_fullname)


# COMMAND ----------

pip install -U langchain-openai

# COMMAND ----------

pip install -U langchain-databricks

# COMMAND ----------

# MAGIC %pip install -U --quiet databricks-sdk==0.29.0

# COMMAND ----------

# MAGIC %pip install -U --quiet langchain-core==0.3.14

# COMMAND ----------

# MAGIC %pip install -U --quiet databricks-vectorsearch==0.40

# COMMAND ----------

# MAGIC %pip install -U --quiet langchain-community==0.2.10

# COMMAND ----------

# MAGIC %pip install -U --quiet typing-extensions==4.12.2

# COMMAND ----------

# MAGIC %pip install -U --quiet mlflow==2.14.3

# COMMAND ----------

# MAGIC %pip install -U --quiet pydantic==2.8.2

# COMMAND ----------

# MAGIC %pip install -U --quiet databricks-sdk==0.29.0 langchain-core==0.3.14 databricks-vectorsearch==0.40 langchain-community==0.2.10 typing-extensions==4.12.2 youtube_search Wikipedia grandalf mlflow==2.14.3 pydantic==2.8.2
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
# from langchain.chat_models import ChatDatabricks
from langchain_databricks.chat_models import ChatDatabricks
# from langchain_openai import ChatOpenAI
from langchain.vectorstores import DatabricksVectorSearch
from databricks.vector_search.client import VectorSearchClient

from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

# Define the Databricks Chat model: DBRX
# import getpass
# import os

# if not os.environ.get("OPENAI_API_KEY"):
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: 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")
llm_dbrx = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens=200, disable_streaming=False)
# llm_dbrx = ChatOpenAI(model="gpt-4o", temperature= 1)
# llm_openai = ChatOpenAI(model="gpt-4", max_tokens=200)
vs_endpoint_name = "cubie_vs_endpoint"
vs_index_table_fullname = "workspace.cubie.customer_embeddings"
# Define the prompt template for Cubie, the chatbot
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


# Define a tool function that the agent will use
# def cubie_chatbot_tool(input_text: str) -> str:
#     # Invoke the chain with the customer's input and retrieve the response
#     response = chain1.invoke({"input": input_text})
#     return response['answer']
# Define a tool function that the agent will use
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


# # Wrap the tool function in a LangChain Tool
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


test_input = "Hi, I'm looking to reserve a storage unit near the 90210 area. Can you help me with that?"

response = cubie_chatbot_tool(test_input)

print("Chatbot response:", response)

# COMMAND ----------

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
import pandas as pd

# Define input and output schema
input_schema = Schema([
    ColSpec("string", "input")  # Assuming the model input is a single string
])
output_schema = Schema([
    ColSpec("string", "answer")  # Assuming the model output is a single string
])

# Create a model signature
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define a custom MLflow model wrapper
class CubieChatbotWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, chain):
        self.chain = chain
    
    def predict(self, context, model_input):
        input_text = model_input["input"].iloc[0]  # Extract single string input
        response = self.chain.invoke({"input": input_text})
        return pd.Series([response['answer']])  # Return as a Pandas Series for MLflow

# Save and log the model with signature
with mlflow.start_run() as run:
    cubie_model = CubieChatbotWrapper(chain=chain1)
    mlflow.pyfunc.log_model(
        "cubie_chatbot_model", 
        python_model=cubie_model, 
        signature=signature,  # Include the signature here
        conda_env="conda.yml"
    )
    model_uri = f"runs:/{run.info.run_id}/cubie_chatbot_model"
    model_version = mlflow.register_model(model_uri, "CubieChatbotModel")


# COMMAND ----------

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

# Define input and output schema
input_schema = Schema([
    ColSpec("string", "input")  # Assuming the model input is a single string
])
output_schema = Schema([
    ColSpec("string", "answer")  # Assuming the model output is a single string
])

# Create a model signature
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define a custom MLflow model wrapper
class CubieChatbotWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, chain):
        self.chain = chain
    
    def predict(self, context, model_input):
        response = self.chain.invoke({"input": model_input})
        return response['answer']

# Save and log the model with signature
with mlflow.start_run() as run:
    cubie_model = CubieChatbotWrapper(chain=chain1)
    mlflow.pyfunc.log_model(
        "cubie_chatbot_model", 
        python_model=cubie_model, 
        signature=signature  # Include the signature here
    )
    model_uri = f"runs:/{run.info.run_id}/cubie_chatbot_model"
    model_version = mlflow.register_model(model_uri, "CubieChatbotModel")

# COMMAND ----------

import mlflow
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModel

class CubieChatbotModel(PythonModel):
    def __init__(self, chat_chain):
        self.chat_chain = chat_chain

    def predict(self, context, model_input):
        return self.chat_chain.invoke({"input": model_input["input"]})['answer']

# Example input and output for signature inference
example_input = {"input": "Can you recommend a storage unit?"}
example_output = {"answer": "Yes, we have units available at 123 Main St."}

# Infer the signature based on example data
signature = infer_signature(model_input=example_input, model_output=example_output)

# Log the model with the inferred signature
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        artifact_path="cubie_chatbot",
        python_model=CubieChatbotModel(chain1),
        signature=signature
    )
    model_uri = f"runs:/{run.info.run_id}/cubie_chatbot"

# COMMAND ----------

model_name = "CubieChatbotModel"
registered_model = mlflow.register_model(model_uri, model_name)

