import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, Table, Column, Integer, String
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool
)
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables (Azure OpenAI keys and SQL connection)
load_dotenv()

# Define Streamlit page settings
st.set_page_config(
    page_title="Sales Data Bot",
    initial_sidebar_state='collapsed',
    layout="wide"
)

# Set the LLM (Language Model)
openai_4_key = os.getenv("openai_4_key")
openai_4_model = os.getenv("openai_4_model")
llm = AzureChatOpenAI(api_key=openai_4_key, model_name=openai_4_model)

# Database connection
engine = create_engine('sqlite:///sakila_master.db')
db = SQLDatabase.from_uri("sqlite:///sakila_master.db")

# Custom SQL database toolkit using LangChain
class CustomSQLDatabaseToolkit(SQLDatabaseToolkit):
    def get_tools(self) -> list:
        """Get the tools for querying the database."""
        # Tool to list tables
        list_tables_tool = ListSQLDatabaseTool(db=self.db)
        
        # Tool to retrieve schema information for the tables
        table_schema_tool = InfoSQLDatabaseTool(
            db=self.db,
            description="Retrieve schema and details of the 'data' table (e.g., Region, Channel, Product, DLR Sales, etc.)"
        )
        
        # Tool to execute SQL queries
        query_sql_database_tool = QuerySQLDataBaseTool(
            db=self.db,
            description="Execute queries based on the provided schema, handling columns like DLR Sales, Region, Channel."
        )
        
        return [list_tables_tool, table_schema_tool, query_sql_database_tool]

# Initialize the toolkit
toolkits = CustomSQLDatabaseToolkit(db=db, llm=llm)

# Create the SQL agent
agent_executors = create_sql_agent(
    llm=llm, toolkit=toolkits, verbose=True, agent_type="openai-functions",
    handle_parsing_errors=True, top_k=10, agent_executor_kwargs={"return_intermediate_steps": True}
)

# Processing the user's query based on Sales.txt schema
def process_user_question(user_question):
    if "dollar growth" in user_question.lower():
        subquery1 = "SELECT Region, Channel, SUM(DLR Sales CHG) as Growth FROM data GROUP BY Region, Channel ORDER BY Growth DESC LIMIT 10"
        subquery2 = "SELECT Region, Channel, SUM(DLR Sales CHG) as Decline FROM data GROUP BY Region, Channel ORDER BY Decline ASC LIMIT 5"
        return [subquery1, subquery2]
    elif "sales by category" in user_question.lower():
        subquery1 = "SELECT Category, SUM(DLR Sales) as TotalSales FROM data GROUP BY Category ORDER BY TotalSales DESC"
        return [subquery1]
    # Add more cases based on schema examples
    return []

# Streamlit user interface
st.markdown("## Sales Data Bot")
user_question = st.text_input("Ask a question about the sales data:")

if st.button("Submit"):
    if user_question:
        queries = process_user_question(user_question)
        if queries:
            for query in queries:
                # Execute each subquery using the SQL agent
                result = agent_executors(query)
                st.write(result["output"])
        else:
            st.write("Sorry, I couldn't process your question. Please ask a question related to sales data.")

# Show database schema (for debugging or user info)
if st.checkbox("Show Table Schema"):
    schema_query = "PRAGMA table_info(data)"
    schema_df = pd.read_sql_query(schema_query, engine)
    st.write(schema_df)

# Hide Streamlit's default header and footer
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)
