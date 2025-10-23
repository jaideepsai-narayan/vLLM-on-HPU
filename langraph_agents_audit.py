# ✅ Smart Compliance Analyzer (Multi-PDF version)
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent
from langchain.tools import Tool
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json, os, glob

# === Agent 1: Multi-Document Loader ===
def load_multiple_documents(folder_path: str) -> str:
    """
    Load and embed multiple PDFs or text files into a single ChromaDB vector store.
    """
    loaders = []
    for file_path in glob.glob(os.path.join(folder_path, "*")):
        if file_path.endswith(".pdf"):
            loaders.append(PyPDFLoader(file_path))
        elif file_path.endswith(".txt"):
            loaders.append(TextLoader(file_path))

    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory="./compliance_db")
    vectordb.persist()

    return f"✅ Loaded {len(chunks)} chunks from {len(loaders)} documents into ChromaDB."

doc_tools = [
    Tool(
        name="LoadMultipleDocuments",
        func=load_multiple_documents,
        description="Load and embed multiple compliance documents from a folder into vector DB"
    )
]

# === Agent 2: Compliance Extraction ===
def extract_compliance_requirements(_: str) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = """
    You are a compliance auditor. From all documents in the vector DB,
    extract and summarize key regulatory or control requirements.
    Output JSON with structure:
    {
      "requirements": [
        {"id": 1, "category": "Data Security", "requirement": "Encrypt sensitive data"},
        {"id": 2, "category": "Access Control", "requirement": "Limit privileged accounts"}
      ]
    }
    """
    return llm.invoke(prompt).content

extract_tools = [
    Tool(name="ExtractComplianceRequirements", func=extract_compliance_requirements,
         description="Extract regulatory and control requirements from multiple documents")
]

# === Agent 3: Audit Analyzer ===
def analyze_audit_findings(requirements_json: str) -> str:
    requirements = json.loads(requirements_json)
    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = f"""
    Given these compliance requirements:
    {requirements}

    Evaluate against audit findings (assume findings from multiple sources are in the DB).
    For each requirement, assess:
    - Compliance status (Compliant / Partial / Non-Compliant)
    - Key gaps and risks
    - Recommended actions
    Return in structured JSON.
    """
    return llm.invoke(prompt).content

analyzer_tools = [
    Tool(name="AnalyzeAuditFindings", func=analyze_audit_findings,
         description="Cross-check audit findings and compliance requirements")
]

# === Agent 4: Report Generator ===
def generate_final_report(audit_json: str) -> str:
    llm = ChatOpenAI(model="gpt-4o")
    prompt = f"""
    Based on this compliance analysis:
    {audit_json}

    Create a final compliance report including:
    1. Executive Summary
    2. Overall Compliance Score
    3. Key Gaps & Risks
    4. Recommended Action Plan
    5. Next Steps for Management Review

    Use markdown formatting.
    """
    return llm.invoke(prompt).content

report_tools = [
    Tool(name="GenerateComplianceReport", func=generate_final_report,
         description="Generate a final markdown compliance report")
]

# === Initialize LLM ===
llm = ChatOpenAI(model="gpt-4o")

# === Create Agents ===
agent1 = create_openai_functions_agent(llm, doc_tools)
agent2 = create_openai_functions_agent(llm, extract_tools)
agent3 = create_openai_functions_agent(llm, analyzer_tools)
agent4 = create_openai_functions_agent(llm, report_tools)

# === Build Graph ===
graph = StateGraph()
graph.add_node("doc_loader", agent1)
graph.add_node("compliance_extractor", agent2)
graph.add_node("audit_analyzer", agent3)
graph.add_node("report_generator", agent4)

# Define pipeline flow
graph.add_edge("doc_loader", "compliance_extractor")
graph.add_edge("compliance_extractor", "audit_analyzer")
graph.add_edge("audit_analyzer", "report_generator")

graph.set_entry_point("doc_loader")

# === Run Pipeline ===
state = graph.invoke({
    "input": "Load all compliance documents from './docs' and produce a unified compliance report."
})

print("\n📘 FINAL MULTI-DOCUMENT COMPLIANCE REPORT:\n")
print(state)
