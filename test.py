import autogen
config_list =[
        {
        "model": "pep-ee-gpt4-1106",
        "api_key": "957badf7e39643dca147a6b8d157f66d",
        "base_url": "https://pep-ee-pepgenxsbx-nonprod-eus2-openai.openai.azure.com/",
        "api_type": "azure",
        "api_version": "2024-02-15-preview"
        }
    ]

llm_config = {"config_list": config_list, "seed": 52,"temperature": 0}







knowledgeAgent_tasks = [
    """You are an expert on Commercial sales and a 6-sigma black-belt certified professional. generate a context of your knowledge""",
    """Please update the given user query: "what is the sales of Extruder in 2023" by replacing the values with the keys from the below table key-desciption table and enhance the query with better phrasing: Year: Timestamp
Month: Timestamp
Week: Timestamp      
Value: Sales Value  
Retailer_Name:  Name of the retailer. e.g: BIG C    
Format: Format/Stores in which sales are made. e.g.: BIGC_DEPOT, BIG C Mini, Food Place, Hypermarket, Market
Manufacturer: Manufacturer of the brands. e.g. PI (Pepsi Cola thai trading), Berli Jucker Food (BJC), Kellogg Thai, Useful, Friendship, United Foods
Brand: Brand of the products. e.g.: Lays, Pringles Chips, Mister Potato, Lay's Stax, Lay;s Max
Subbrand: Subbrand names of the products e.g.: Lays, Pringles Chips, Mister Potato, Lay's Stax, Lay;s Max
Segment: Segment for the products   e.g.: Cfish (Cuttle Fish), Corn, Extruder (EXT), Fish (Fish Snack), MEAT, PC (Potato Chip), PNUT (Peanut), POP (Popcorn), Prawn (Prawn Crackers), RC (Rice Cracker)
SubSegment: Subsegment for the products e.g: Natural, Fabricated
Category: whether the product is savoury (SAV) or non - savoury  
Pack_Size: the size of the products meaured in baths eg - 10B, 100B  
Pack_Type: type of the packaging of the product - eg SG (single pack) or BP (bundle pack)    
Basesize: Size of the product in grams  
Package: Type of displaying the products in markets - Bowl, cup, tray, canister  
End_Report_Format: a column to match the format of the end report - meaning same as format  """,
]
 
writing_tasks = ["""Rephrase the user query: "what is the sales of ext in 2023"""]
 
function_knowledge_assistant = autogen.AssistantAgent(
    name="Financial_assistant",
    llm_config=llm_config,
)
schema_assistant = autogen.AssistantAgent(
    name="Researcher",
    llm_config=llm_config,
)
writer = autogen.AssistantAgent(
    name="writer",
    llm_config=llm_config,
    system_message="""
        You are a professional writer, known for
        your insightful queries.
        Reply "TERMINATE" in the end when everything is done.
        """,
)
 
user = autogen.UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "last_n_messages": 1,
        "work_dir": "tasks",
        "use_docker": False,
    },  
)
 
chat_results = user.initiate_chats(
    [
        {
            "recipient": schema_assistant,
            "message": knowledgeAgent_tasks[1],
            "summary_method": "reflection_with_llm",
        },
        {
            "recipient": writer,
            "message": writing_tasks[0],
            "carryover": "I want to include a figure or a table of data in the blogpost.",
        },
    ]
)
 


data = ''
for i, chat_res in enumerate(chat_results):
  data = chat_res.chat_history


Level1_Question = data[1]["content"].split("\n")[0]
print(Level1_Question)

#     # Skip the initial message from the user
#     if i == 0 and 'You are an expert on Commercial sales' in chat_res['message']:
#         continue

#     # Extract final response
#     final_response = chat_res['message']
#     if final_response.strip() != 'TERMINATE':
#         print(f"Final Result from chat {i + 1}: {final_response.strip()}")

      
# for i, chat_res in enumerate(chat_results):
#     # print(f"*****{i}th chat*******:")
#     print(chat_results.get('final_response', {}).get('content', ''))
#     # print(chat_res[len(chat_res)-1].summary)
#     # if chat_res.get('name') == 'assistant' and chat_res['content'] != '':
#     #   print("Human input in the middle:", chat_res.chat_history)
#     # # print("Conversation cost: ", chat_res.cost)
#     # print("\n\n")
    
# #     final_message = chat_res.messages[-1] if chat_res.messages else None
# #     if final_message:
# #         print(f"Finalized Result from chat {i + 1}: {final_message.get('content', '').strip()}")
# #     print("\n\n")

# # for i, chat_res in enumerate(chat_results):
# #   if chat_res.get('name') == 'Researcher' and chat_res['content'] != '':
# #     data_content = f"{chat_res['content']}\n\n"
# #     print("Output:" + data_content)