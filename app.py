import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from langchain.prompts.chat import ChatPromptTemplate
from langchain.agents import AgentType
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.prompts.chat import ChatPromptTemplate
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
import os
from langchain.agents import *
from langchain_community.llms import OpenAI
from langchain.agents.agent_types import AgentType
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain_community.tools import BaseTool
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)
from PIL import Image
from typing import List, Dict
from langchain_community.tools import BaseTool
from langchain_openai import AzureChatOpenAI
import pandas as pd
from pandasai import SmartDataframe
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback
import time
from datetime import datetime
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache
import azure.cognitiveservices.speech as speechsdk



import os
import streamlit as st
from openai import OpenAI
import base64
from utils import get_image_description




# Azure Speech Service configuration
speech_key = "6555eeb2e3a34f8e9fec52bef46c819d"
service_region = "eastus"


global Prompt_Tokens, Completion_Tokens ,Total_Cost , processing_time

st.session_state.Prompt_Tokens = 0
st.session_state.Completion_Tokens = 0
st.session_state.Total_Cost = 0
st.session_state.processing_time = 0

# Set your Azure Cognitive Services subscription key and region
subscription_key = '6555eeb2e3a34f8e9fec52bef46c819d'
service_region = 'eastus'

def convert_text_to_speech(text):
    try:
        speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=service_region)
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
        result = speech_synthesizer.speak_text_async(text).get()
        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            st.error(f"Error: {result.reason}")
    except Exception as e:
        st.error(f"Speech synthesis error: {e}")

# Custom toolkit for SQL database interaction
def burn_tokens(prompt_tokens, completion_tokens, total_cost):
    # global Prompt_Tokens, Completion_Tokens ,Total_Cost
    st.session_state.Prompt_Tokens += prompt_tokens
    st.session_state.Completion_Tokens += completion_tokens
    st.session_state.Total_Cost += total_cost
    # st.write(str(Prompt_Tokens) + ":" + str(Completion_Tokens) + ":" + str(Total_Cost))

class CustomSQLDatabaseToolkit(SQLDatabaseToolkit):
    def get_tools(self) -> list[BaseTool]:
        """Get the tools in the toolkit."""
        list_tables_tool = ListSQLDatabaseTool(db=self.db)
        table_schema_tool_description = (
            "Use this tool to get the schema of specific tables. "
            "Input to this tool is a comma-separated list of tables, output is the "
            "schema and sample rows for those tables. "
            "Be sure that the tables actually exist by calling "
            f"{list_tables_tool.name} first! "
            "Example Input: table1, table2, table3"  # NOTE: removed the single quotes here
        )
        table_schema_tool = InfoSQLDatabaseTool(
            db=self.db,
            description=table_schema_tool_description,
        )
        query_sql_database_tool_description = (
            "Use this tool to execute query and get result from the database. "
            "Input to this tool is a SQL query, output is a result from the database. "
            "If the query is not correct, an error message will be returned. "
            "If an error is returned, rewrite the query and try again. "
            "If you encounter an issue with Unknown column "
            f"'xxxx' in 'field list', or no such column 'xxxx', use {table_schema_tool.name} "
            "to get the correct table columns."
        )
        query_sql_database_tool = QuerySQLDataBaseTool(
            db=self.db,
            description=query_sql_database_tool_description,
        )
        query_sql_checker_tool_description = (
            "Use this tool to check if your query is correct before executing "
            "it. Always use this tool before executing a query with "
            f"{query_sql_database_tool.name}!"
        )
        query_sql_checker_tool = QuerySQLCheckerTool(
            db=self.db,
            llm=self.llm,
            description=query_sql_checker_tool_description,
        )
        return [
            list_tables_tool,
            table_schema_tool,
            query_sql_checker_tool,
            query_sql_database_tool,
        ]


sql_icon = "\U0001F4DD"  # Memo for "SQL query"
st.set_page_config(
    page_title="Data Bot",
    initial_sidebar_state='collapsed',
    page_icon=sql_icon,
    layout="wide")

tab1, tab2 = st.tabs(["Data Bot", "Vision Bot"])
gpt4 = st.toggle('GPT 4')

# Content for Tab 11
with tab1:
    colh1, colh2, colh3 = st.columns((7, 1.5, 1.5))
    with colh1:        
        if gpt4:
            llm = AzureChatOpenAI(
                    openai_api_version="2024-02-15-preview",
                    azure_deployment="pep-ee-gpt4-1106",
                    openai_api_key="957badf7e39643dca147a6b8d157f66d",
                    azure_endpoint="https://pep-ee-pepgenxsbx-nonprod-eus2-openai.openai.azure.com/",
                    model="pep-ee-gpt4-1106"
                )
        else:
            llm = ChatOpenAI(api_key="sk-proj-WWg7S2Zmsl9fFRcxRxOfT3BlbkFJLwicDAKfRcixnNyhPLDf",model_name="gpt-4o-2024-05-13",temperature=0.1,max_retries=5,max_tokens=2000)

    def update_values(Prompt_Tokens,Completion_Tokens,Total_Cost,processing_time):
        with colh2:
            # update_values()
            data = {
                    "Metric": ["Prompt Tokens", "Completion Tokens"],
                    "Value": [Prompt_Tokens, Completion_Tokens]
                }
            df1 = pd.DataFrame(data)
            st.write(df1.to_html(index=False, header=False, escape=False), unsafe_allow_html=True)
            st.markdown("""
            <style>
                table {
                    font-size: 0.6rem;  /* Adjust font size */
                    width: 90%;         /* Adjust table width */
                    white-space: nowrap;
                }

                th, td {
                    padding: 10px;       /* Reduce cell padding */
                }
                .container {
                        display: flex;
                        justify-content: flex-end;
                    }
            </style>
            """, unsafe_allow_html=True)
            st.write('</div>', unsafe_allow_html=True)
        with colh3:
            # processing_time1 = st.session_state.processing_time
            # processing_time1 = 1
            data1 = {
                    "Metric": ["Total Cost", "Processing Time"],
                    "Value": [Total_Cost, processing_time]
                }
            
            df2 = pd.DataFrame(data1)
            st.write(df2.to_html(index=False, header=False, escape=False), unsafe_allow_html=True)
            st.markdown("""
            <style>
                table {
                    font-size: 0.6rem;  /* Adjust font size */
                    width: 90%;         /* Adjust table width */
                    white-space: nowrap;
                }

                th, td {
                    padding: 5px;       /* Reduce cell padding */
                }
                .container {
                        display: flex;
                        justify-content: flex-end;
                    }
            </style>
            """, unsafe_allow_html=True)
            st.write('</div>', unsafe_allow_html=True)
    # Database connection
    engine = create_engine('sqlite:///sakila_master.db')
    db = SQLDatabase.from_uri("sqlite:///sakila_master.db")
    toolkits = CustomSQLDatabaseToolkit(db=db, llm=llm)

    agent_executors = create_sql_agent(
                llm=llm, toolkit=toolkits, verbose=True, agent_type="openai-functions",
            handle_parsing_errors=True, top_k=10, agent_executor_kwargs={"return_intermediate_steps": True}
        )


    st.write(
        f'<div style="display: flex; align-items: center;"><span style="font-size: 36px;">\U0001F4DD</span><h1 style="margin-left: 10px;">Data Bot</h1></div>',
        unsafe_allow_html=True
    )

    # Load your image (replace 'your_image.png' with the path to your image)
    image = Image.open('Sakiladb.png')
    
    # Create an expander
    with st.expander("See explanation and image"):
    # Display the image within the expander
        st.image(image, caption='Image Caption', use_column_width=True)


    with st.expander("Interactive DB network visualizations"):
    # Display the image within the expander
        import sqlite3
        from pyvis.network import Network
        conn = sqlite3.connect('sakila_master.db')
        cursor = conn.cursor()

        # Get table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        tables = [table[0] for table in tables]

        # Get record count for each table
        table_counts = {}
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table};")
            count = cursor.fetchone()[0]
            table_counts[table] = count

        # Get foreign key relationships
        foreign_keys = {}
        for table in tables:
            cursor.execute(f"PRAGMA foreign_key_list({table});")
            foreign_key_info = cursor.fetchall()
            foreign_tables = set([info[2] for info in foreign_key_info])
            foreign_keys[table] = foreign_tables

        # Close database connection
        conn.close()

        # Create a pyvis network
        net = Network()

        # Add nodes for each table
        for table in tables:
            count = table_counts.get(table, 0)
            net.add_node(table, label=f"{table} ({count})", shape='circle',group=3)

        # Add edges for foreign key relationships
        for source_table, target_tables in foreign_keys.items():
            for target_table in target_tables:
                net.add_edge(source_table, target_table)

        # Write HTML output to file
        output_file = 'database_schema.html'
        net.write_html(output_file)
        # st.image(image, caption='Image Caption', use_column_width=True)
        st.components.v1.html(open(output_file, 'r').read(), width=1000, height=700, scrolling=False)


    thinking_face_emoji = "\U0001F914"

    # Unicode representations of selected emojis
    database_emoji = "\U0001F4BE"  # Floppy disk for "database"
    agents_emoji = "\U0001F575"    # Detective for "agents"
    sql_query_emoji = "\U0001F4DD" # Memo for "SQL query"
    final_result_emoji = "\u2705"
    achievement_emoji = "\U0001F3C6"


    st.write("You are a helpful AI assistant expert in querying SQL Database to find answers to user's questions about actor, address, and category.")
    # Wrap your input and submit button in a form
    with st.form("user_query_form"):            
        user_question = st.text_input("Ask me a question:")
        submit_button = st.form_submit_button("Submit")
        st.write("*Add - at the end of your prompt and mention your specify the type of chart you like present.")
    # working Code
    st.session_state.processing_time = 0
    col1, col2, col3 = st.columns((4, 3, 3))
    try:
        UserQuestion = user_question.split("-")[0]
        CharType = user_question.split("-")[1]
    except Exception as e:
        UserQuestion = user_question.split("-")[0]
        CharType = ""
    if submit_button and UserQuestion:
        start_time = time.perf_counter()
        prompt_tokens = 0
        completion_tokens = 0
        total_cost = 0
        load_dotenv()
        with get_openai_callback() as cb:
            result = agent_executors(UserQuestion)  # Make sure this function is defined to return the expected structure
            prompt_tokens = prompt_tokens + cb.prompt_tokens
            completion_tokens = completion_tokens + cb.completion_tokens
            total_cost = total_cost + cb.total_cost
            # st.write(cb)
        # Output styled_text
        styled_text = f'<b style="font-weight: 1000; font-size: 25px;">Insight</b>'
        with col1:
            st.write(styled_text, unsafe_allow_html=True)
            with get_openai_callback() as cb:
                response = result["output"]
                st.write(response)
                prompt_tokens = prompt_tokens + cb.prompt_tokens
                completion_tokens = completion_tokens + cb.completion_tokens
                total_cost = total_cost + cb.total_cost
        # Output styled_text1
        styled_text1 = f'<b style="font-weight: 1000; font-size: 25px;">AI Agents Group Chat</b>'
        # with col3:
            
    
            # Attempt to execute the last SQL query from the interaction (if applicable)
        if result.get("intermediate_steps"):
            with st.expander("Intermediate Steps"):
                st.sidebar.write(styled_text1, unsafe_allow_html=True)
                for step in result["intermediate_steps"]:
                    with get_openai_callback() as cb:
                        action, results = step
                        query = action.tool_input
                        st.sidebar.markdown(
                            f"""
                            <div style="position: relative; border-radius: 10px; border: 2px solid #807e7e; background-color: #807e7e; padding: 10px; color: white; margin-bottom: 10px;">
                                <div style="font-weight: bold;">{database_emoji} SQL Database Agent:</div>
                                <div>{action.tool}</div>
                                <div style="position: absolute; bottom: -10px; left: 10px; width: 0; height: 0; border-style: solid; border-width: 10px 10px 0 10px; border-color: #807e7e transparent transparent transparent;"></div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        st.sidebar.markdown(
                            f"""
                            <div style="position: relative; border-radius: 10px; border: 2px solid #807e7e; background-color: #807e7e; padding: 10px; color: white; margin-bottom: 10px;">
                                <div style="font-weight: bold;">{agents_emoji} SQL Query Agent:</div>
                                <div>{action.tool_input}</div>
                                <div style="position: absolute; bottom: -10px; left: 10px; width: 0; height: 0; border-style: solid; border-width: 10px 10px 0 10px; border-color: #807e7e transparent transparent transparent;"></div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        st.sidebar.markdown(
                            f"""
                            <div style="position: relative; border-radius: 10px; border: 2px solid #4CAF50; background-color: #4CAF50; padding: 10px; text-align: right; color: white; margin-bottom: 10px;">
                                <div style="font-weight: bold;">{final_result_emoji} Final Result:</div>
                                <div>{results}</div>
                                <div style="position: absolute; bottom: -10px; right: 10px; width: 0; height: 0; border-style: solid; border-width: 10px 10px 0 10px; border-color: #4CAF50 transparent transparent transparent;"></div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        prompt_tokens = prompt_tokens + cb.prompt_tokens
                        completion_tokens = completion_tokens + cb.completion_tokens
                        total_cost = total_cost + cb.total_cost
                        # st.write(cb)
                        

                    # st.markdown("---")
        else:
            st.sidebar.write("No intermediate steps found in the result.")


        # try:
            # Output styled_text2
        styled_text2 = f'<b style="font-weight: 1000; font-size: 25px;">Table</b>'
        with col2:
            try:
                with get_openai_callback() as cb:    
                    st.write(styled_text2, unsafe_allow_html=True)
                    df = pd.read_sql_query(query, engine)
                    st.write(df)
                    prompt_tokens = prompt_tokens + cb.prompt_tokens
                    completion_tokens= completion_tokens + cb.completion_tokens
                    total_cost = total_cost + cb.total_cost
                    # st.write(cb)
            
            except Exception as e:
                st.write("The absence of a dataframe is due to the nature of the question, which is categorized as a general inquiry")
                        
    
        with col3:
            styled_text3 = f'<b style="font-weight: 1000; font-size: 25px;">Chart</b>'
            st.write(styled_text3, unsafe_allow_html=True)
            try:
                
                #Chart PandasAI
                # os.remove("/home/DFCAI/AOPV1test/exports/charts/temp_chart.png")
                with get_openai_callback() as cb:
                    sdf = SmartDataframe(df, config={"llm": llm})
                    chartpath = sdf.chat(user_question + CharType + " Create a chart using the data frame")
                    # st.write(chartpath)
                    st.image(chartpath, caption='Visualization')
                    prompt_tokens = prompt_tokens + cb.prompt_tokens
                    completion_tokens = completion_tokens + cb.completion_tokens
                    total_cost = total_cost + cb.total_cost
                    # st.write(cb)
            except Exception as e:
                st.write("You don't have any graph for this or dataframe for this as it's a normal summary or insights question.")
        end_time = time.perf_counter()
        
        st.session_state.processing_time = end_time - start_time
        # st.write(str(prompt_tokens),str(completion_tokens),str(total_cost),str(st.session_state.processing_time))
        update_values(str(prompt_tokens),str(completion_tokens),str(total_cost),str(st.session_state.processing_time))
    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}  # Hide main menu
                header { visibility: hidden; }  # Hide entire header
                footer {visibility: hidden;} # Hide footer (optional)
                
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)


with tab2:

    # Textbox for updating OpenAI API key
    api_key = st.text_input("Enter your OpenAI API key", type="password")
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY", "")

    if api_key:
        # Initialize the OpenAI client
        client = OpenAI(api_key=api_key)

        # Textbox for updating the prompt
        prompt = st.text_input("Enter the prompt for image description", "What’s in this image?")

        # Upload image button
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            try:
                # Display the uploaded image
                st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
                st.write("")
                st.write("Classifying...")

                # Get the image description
                description = get_image_description(client, uploaded_file, prompt)
                st.write(description)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.error("Please provide a valid OpenAI API key.")
