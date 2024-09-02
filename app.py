import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, Table, Column, Integer, String
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
# from pandasai import SmartDataframe
from langchain_openai import ChatOpenAI
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
import uuid


from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import streamlit as st
import azure.cognitiveservices.speech as speechsdk
import speech_recognition as sr

import streamlit.components.v1 as components

from datetime import datetime
import autogen

# import sounddevice as sd
# import wave
# import numpy as np

# Azure Speech Service configuration




unique_id = str(uuid.uuid4())
Base = declarative_base()

class ProjectMetadata(Base):
    __tablename__ = 'Project_Metadata'  # Matches your table name

    # Define columns according to the schema
    Timestamp = Column(String)
    Prompt_Tokens = Column(Integer)
    Completion_Tokens = Column(Integer)
    Total_Cost = Column(Integer)
    Processing_Time = Column(Integer)
    Model = Column(String)
    uniquid = Column(Integer, primary_key=True)








#Load .env
from dotenv import load_dotenv
import os
load_dotenv() 


global Prompt_Tokens, Completion_Tokens ,Total_Cost , processing_time

st.session_state.Prompt_Tokens = 0
st.session_state.Completion_Tokens = 0
st.session_state.Total_Cost = 0
st.session_state.processing_time = 0

# Set your Azure Cognitive Services subscription key and region
speech_key = os.getenv("Azure_Cognitive_Services")
service_region = os.getenv("Azure_Cognitive_Region")

#  GPT 4 1106 key
openai_4_version = os.getenv("openai_4_version"),
openai_4_key = os.getenv("openai_4_key"),
openai_4_deployment = os.getenv("openai_4_deployment"),
openai_4_endpoint = os.getenv("openai_4_endpoint"),
openai_4_model = os.getenv("openai_4_model")

# GPT 4o key
openai_4o_key = os.getenv("openai_4o_key")
openai_4o_model = os.getenv("openai_4o_model")

def convert_text_to_speech(text):
    try:
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
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




st.markdown("""
    <style>
    .stRadio > div {flex-direction: row;}
    </style>
    """, unsafe_allow_html=True)

# Options for the radio buttons
options = ["More Creative", "More Balanced", "More Precise"]

# Radio button widget
selected_option = st.sidebar.radio("Choose an option:", options)

# Set the temperature based on the selected option
if selected_option == "More Creative":
    temperature_op = 0.9
elif selected_option == "More Balanced":
    temperature_op = 0.5
else:
    temperature_op = 0.2


User_Persona = st.sidebar.selectbox('Select your Persona:',('Sales Manager', 'Sales Chief', 'Product Category'))
# st.write(temperature_op)




Access_User = st.sidebar.toggle("Enable Report",value=False)

if Access_User:
    tab1, tab2, tab3, tab4 = st.tabs(["Data Bot", "Vision Bot", "Voice Bot", "Report"])
else:
    tab1, tab2 = st.tabs(["Data Bot", "Vision Bot"])

# Content for Tab 11
with tab1:
    colh1, colh2, colh3 = st.columns((7, 1.5, 1.5))
    with colh1:
        # value = st_toggle_switch("Select Value", options=["Option 1", "Option 2"], initial_value=0)
        
        
        # gpt4 = st.sidebar.toggle('',value=True)
        

        GPT_options = ["GPT-4 1106", "Using GPT-4o"]
        selected_LLM = st.sidebar.radio("Choose an option:", GPT_options)
        if selected_LLM == "GPT-4 1106":
            gpt4 = True
        else:
            gpt4 = False

        

        
        if gpt4:
            st.write("Using GPT-4 1106")
            llm = AzureChatOpenAI(openai_api_version=os.getenv("openai_4_version"),azure_deployment=os.getenv("openai_4_deployment"),openai_api_key=os.getenv("openai_4_key"),azure_endpoint=os.getenv("openai_4_endpoint"),model=os.getenv("openai_4_model"),temperature=temperature_op)
        else:
            st.write("Using GPT-4o")
            # llm = ChatOpenAI(api_key=openai_4o_key,model_name=openai_4o_model,temperature=0.1,max_retries=5,max_tokens=2000)
            llm = ChatOpenAI(api_key=openai_4o_key,model_name="gpt-4o-2024-05-13",temperature=temperature_op,max_retries=5,max_tokens=2000)

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
    # engine = create_engine('sqlite:///sakila_master.db', connect_args={'busy_timeout': 10000})  # Wait for 10 seconds (adjust timeout value)
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

    # # Load your image (replace 'your_image.png' with the path to your image)
    # image = Image.open('Sakiladb.png')
    
    # # Create an expander
    # with st.expander("See explanation and image"):
    # # Display the image within the expander
    #     st.image(image, caption='Image Caption', use_column_width=True)


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
        #st.components.v1.html(open(output_file, 'r').read(), width=1000, height=700, scrolling=False)


    thinking_face_emoji = "\U0001F914"

    # Unicode representations of selected emojis
    database_emoji = "\U0001F4BE"  # Floppy disk for "database"
    agents_emoji = "\U0001F575"    # Detective for "agents"
    sql_query_emoji = "\U0001F4DD" # Memo for "SQL query"
    final_result_emoji = "\u2705"
    achievement_emoji = "\U0001F3C6"

    
    st.write("You are a helpful AI assistant expert in querying SQL Database to find answers to user's questions.")
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
        # st.write("Your unique ID is:", unique_id)
        # st.write(str(prompt_tokens))
        # st.write(str(completion_tokens))
        # st.write(str(total_cost))
        # st.write(str(st.session_state.processing_time))


        
        # DB Update
        # Create a session
        Session = sessionmaker(bind=engine)
        session = Session()
        # Insert data (replace with your actual values)
        if gpt4:
            using_mod="pep-ee-gpt4-1106"
        else:
            using_mod="gpt-4o-2024-05-13"

        new_project_metadata = ProjectMetadata(
            Timestamp=str(datetime.now()),  # Assuming you have a timestamp string
            Prompt_Tokens=str(prompt_tokens),
            Completion_Tokens=str(completion_tokens),
            Total_Cost=str(total_cost),
            Processing_Time=str(st.session_state.processing_time),
            uniquid=str(unique_id),
            Model=str(using_mod)
        )
        session.add(new_project_metadata)

        # Commit the changes to the database
        session.commit()

        # Close the session
        session.close()


    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}  # Hide main menu
                header { visibility: hidden; }  # Hide entire header
                footer {visibility: hidden;} # Hide footer (optional)
                
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)











######################################################################################################################################################################################################################3
with tab2:

    st.write(
        f'<div style="display: flex; align-items: center;"><span style="font-size: 36px;">\U0001F4DD</span><h1 style="margin-left: 10px;">Vision Bot</h1></div>',
        unsafe_allow_html=True
    )
    start_time = 0
    end_time = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_cost = 0
    # Textbox for updating OpenAI API key
    # api_key = "sk-proj-WWg7S2Zmsl9fFRcxRxOfT3BlbkFJLwicDAKfRcixnNyhPLDf"
    if not openai_4o_key:
        api_key = openai_4o_key

    if openai_4o_key:
        # Initialize the OpenAI client
        api_key = openai_4o_key
        client = OpenAI(api_key=api_key)

        # Textbox for updating the prompt
        prompt = st.text_input("Enter the prompt for image description", "What’s in this image?")

        # Upload image button
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            try:
                start_time = time.perf_counter()
                # Display the uploaded image
                st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
                st.write("")
                st.write("Classifying...")

                with get_openai_callback() as cb:
                    # Get the image description
                    description = get_image_description(client, uploaded_file, prompt)
                    st.write(description)
                    end_time = time.perf_counter()
                    st.session_state.processing_time = end_time - start_time
                    prompt_tokens = prompt_tokens + cb.prompt_tokens
                    completion_tokens = completion_tokens + cb.completion_tokens
                    total_cost = total_cost + cb.total_cost
                    update_values(str(prompt_tokens),str(completion_tokens),str(total_cost),str(st.session_state.processing_time))
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.error("Please provide a valid OpenAI API key.")
    
    

    # DB Update
    # Create a session
    Session = sessionmaker(bind=engine)
    session = Session()
    # Insert data (replace with your actual values)
    using_mod="gpt-4o-2024-05-13"

    new_project_metadata = ProjectMetadata(
        Timestamp=str(datetime.now()),  # Assuming you have a timestamp string
        Prompt_Tokens=str(prompt_tokens),
        Completion_Tokens=str(completion_tokens),
        Total_Cost=str(total_cost),
        Processing_Time=str(st.session_state.processing_time),
        uniquid=str(unique_id),
        Model=str(using_mod)
    )
    session.add(new_project_metadata)

    # Commit the changes to the database
    session.commit()

    # Close the session
    session.close()






















###########################################################################################################################################################################################################################
if Access_User:
    with tab3:
        # Read the local HTML file
        # html_file_path = "input.html"
        # with open(html_file_path, 'r', encoding='utf-8') as html_file:
        #     html_content = html_file.read()

        # # Display the HTML file in the Streamlit app
        # components.html(html_content, height=600, scrolling=False)
        


        colh1, colh2, colh3 = st.columns((7, 1.5, 1.5))
        with colh1:
            gpt41 = st.toggle(' ',value=True)
            if gpt4:
                # llm = AzureChatOpenAI(openai_api_version=openai_4_version, azure_deployment=openai_4_deployment, openai_api_key=openai_4_key, azure_endpoint=openai_4_endpoint, model=openai_4_model)
                st.write("Using GPT-4 1106")
                llm = AzureChatOpenAI(
        openai_api_version=os.getenv("openai_4_version"),
        azure_deployment=os.getenv("openai_4_deployment"),
        openai_api_key=os.getenv("openai_4_key"),
        azure_endpoint=os.getenv("openai_4_endpoint"),
        model=os.getenv("openai_4_model")
    )
            else:
                st.write("Using GPT-4o")
                llm = ChatOpenAI(api_key=openai_4o_key,model_name=openai_4o_model,temperature=0.1,max_retries=5,max_tokens=2000)

        def update_values1(Prompt_Tokens,Completion_Tokens,Total_Cost,processing_time):
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
            f'<div style="display: flex; align-items: center;"><span style="font-size: 36px;">\U0001F4DD</span><h1 style="margin-left: 10px;">Voice Bot</h1></div>',
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

        thinking_face_emoji = "\U0001F914"

        # Unicode representations of selected emojis
        database_emoji = "\U0001F4BE"  # Floppy disk for "database"
        agents_emoji = "\U0001F575"    # Detective for "agents"
        sql_query_emoji = "\U0001F4DD" # Memo for "SQL query"
        final_result_emoji = "\u2705"
        achievement_emoji = "\U0001F3C6"


        st.write("You are a helpful AI assistant expert in querying SQL Database to find answers to user's questions.")

        def speech_to_text_from_microphone1():
            speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
            audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
            speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
            try:
                st.info("Recording audio using microphone. Please speak clearly.")
                result = speech_recognizer.recognize_once_async().get()
                return result.text
            except Exception as e:
                return f"Error: {str(e)}"


        if 'user_question' not in st.session_state:
            st.session_state.user_question1 = ""
        user_question_input1 = st.empty()
        record_button = st.button("🎙️")

        
        with st.form("user_query_form1"):            
            st.session_state.user_question1 = user_question_input1.text_input("Ask me a question:", st.session_state.user_question1)
            user_question = st.text_input("Ask me a question:")
            submit_button = st.form_submit_button("Submit")
            st.write("*Add - at the end of your prompt and mention your specify the type of chart you like present.")
        

        if record_button:
            transcription = speech_to_text_from_microphone1()
            if transcription:
                # Update user_question with transcribed text
                st.session_state.user_question1 = transcription
                # user_question = transcription
                # Update the text input field with the transcribed text
                user_question_input1.text_input("Your question here:", st.session_state.user_question1)

        
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
            update_values1(str(prompt_tokens),str(completion_tokens),str(total_cost),str(st.session_state.processing_time))
        hide_st_style = """
                    <style>
                    #MainMenu {visibility: hidden;}  # Hide main menu
                    header { visibility: hidden; }  # Hide entire header
                    footer {visibility: hidden;} # Hide footer (optional)
                    
                    </style>
                    """
        st.markdown(hide_st_style, unsafe_allow_html=True)

################################################################################

    with tab4:

        st.write(
            f'<div style="display: flex; align-items: center;"><span style="font-size: 36px;">\U0001F4DD</span><h1 style="margin-left: 10px;">Log Report</h1></div>',
            unsafe_allow_html=True
        )

        # Create a session
        Session = sessionmaker(bind=engine)
        session = Session()

        records = session.query(ProjectMetadata).all()
        data = {
            'uniquid': [record.uniquid for record in records],
            'Timestamp': [record.Timestamp for record in records],
            'Prompt_Tokens': [record.Prompt_Tokens for record in records],
            'Completion_Tokens': [record.Completion_Tokens for record in records],
            'Total_Cost': [record.Total_Cost for record in records],
            'Processing_Time': [record.Processing_Time for record in records],
            'Model': [record.Model for record in records]
        }
        df = pd.DataFrame(data)
        st.write(df)