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

