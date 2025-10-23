# ✅ LangGraph Agents for Medicine Inventory Management

from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent
from langchain.tools import Tool
import sqlite3
import pandas as pd
from datetime import datetime
import json


# === Agent 1: Database Manager ===
def get_inventory_data(_: str) -> str:
    conn = sqlite3.connect("medicine_inventory.db")
    query = "SELECT * FROM medicine"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df.to_json(orient="records")

def add_new_medicine(medicine_json: str) -> str:
    data = json.loads(medicine_json)
    conn = sqlite3.connect("medicine_inventory.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO medicine (name, content, manufacture_date, expiry_date, stock, price)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (data['name'], data['content'], data['manufacture_date'], data['expiry_date'], data['stock'], data['price']))
    conn.commit()
    conn.close()
    return f"Added new medicine: {data['name']}"

db_tools = [
    Tool(name="GetInventoryData", func=get_inventory_data, description="Fetch all medicines from database"),
    Tool(name="AddNewMedicine", func=add_new_medicine, description="Add new medicine record to the database")
]


# === Agent 2: Inventory Analyzer ===
def analyze_inventory(data_json: str) -> str:
    df = pd.DataFrame(json.loads(data_json))
    today = datetime.now().date()

    expired = df[pd.to_datetime(df["expiry_date"]).dt.date < today]
    low_stock = df[df["stock"] < 20]

    analysis = {
        "expired_count": len(expired),
        "low_stock_count": len(low_stock),
        "expired_medicines": expired.to_dict(orient="records"),
        "low_stock_medicines": low_stock.to_dict(orient="records")
    }
    return json.dumps(analysis)

analysis_tools = [
    Tool(name="AnalyzeInventory", func=analyze_inventory, description="Find expired and low-stock medicines")
]


# === Agent 3: Report Generator ===
def generate_report(analysis_json: str) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = f"""
    You are a pharmacy assistant. Analyze this medicine inventory analysis:
    {analysis_json}

    Write a short report that:
    1. Summarizes current inventory issues.
    2. Lists expired or soon-to-expire medicines.
    3. Suggests restocking or removal actions.
    Format clearly and professionally.
    """
    return llm.invoke(prompt).content

report_tools = [
    Tool(name="GenerateReport", func=generate_report, description="Summarize inventory analysis into human-readable form")
]


# === Initialize Agents ===
llm = ChatOpenAI(model="gpt-4o")

agent1 = create_openai_functions_agent(llm, db_tools)
agent2 = create_openai_functions_agent(llm, analysis_tools)
agent3 = create_openai_functions_agent(llm, report_tools)

# === Build LangGraph Workflow ===
graph = StateGraph()

graph.add_node("db_agent", agent1)
graph.add_node("inventory_analyzer", agent2)
graph.add_node("report_generator", agent3)

# Define workflow: DB → Analysis → Report
graph.add_edge("db_agent", "inventory_analyzer")
graph.add_edge("inventory_analyzer", "report_generator")

# Entry point
graph.set_entry_point("db_agent")

# === Run End-to-End Workflow ===
state = graph.invoke({
    "input": "Analyze the current medicine inventory and generate a report."
})

print("\n💊 FINAL INVENTORY HEALTH REPORT:\n")
print(state)
