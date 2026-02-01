from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from reports.report_generator import generate_pdf_report
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from reports.report_generator import generate_pdf_report



from autogen import AssistantAgent, UserProxyAgent
from autogen.agentchat import GroupChat, GroupChatManager

# Global storage
RAW_DATASET = None
CLEAN_DATASET = None
FEATURE_DATASET = None
CLEANING_REPORT = ""
FEATURE_REPORT = ""
VISUAL_REPORT = ""

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="AutoGen Data Analyzer Backend")


class AnalyzeRequest(BaseModel):
    query: str


# -------------------------
# Ollama config
# -------------------------
llm_config = {
    "config_list": [
        {
            "model": "llama3",
            "api_key": "ollama",
            "base_url": "http://localhost:11434/v1",
        }
    ],
    "temperature": 0.2,
}


# -------------------------
# Data Cleaning
# -------------------------
def clean_dataset(df):
    report = []
    original_rows = len(df)
    df = df.drop_duplicates()
    report.append(f"Removed {original_rows - len(df)} duplicate rows")

    missing_before = df.isnull().sum().sum()
    df = df.fillna(method="ffill").fillna(method="bfill")
    missing_after = df.isnull().sum().sum()
    report.append(f"Missing values before: {missing_before}, after: {missing_after}")

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    report.append("Standardized column names")

    return df, "\n".join(report)


# -------------------------
# Feature Engineering
# -------------------------
def engineer_features(df):
    report = []

    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col])
            df["year"] = df[col].dt.year
            df["month"] = df[col].dt.month
            report.append(f"Extracted year/month from {col}")
            break
        except:
            continue

    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        df[f"{col}_rolling_avg"] = df[col].rolling(3).mean()
        report.append(f"Added rolling average for {col}")

    return df, "\n".join(report)


# -------------------------
# Visualizer
# -------------------------
def generate_visuals(df):
    report = []

    numeric_cols = df.select_dtypes(include="number").columns

    if len(numeric_cols) == 0:
        return "No numeric columns available for visualization."

    # Line plot
    plt.figure()
    df[numeric_cols[0]].plot(title=f"{numeric_cols[0]} Trend")
    line_path = os.path.join(OUTPUT_DIR, "line_chart.png")
    plt.savefig(line_path)
    plt.close()
    report.append(f"Line chart saved: {line_path}")

    # Histogram
    plt.figure()
    df[numeric_cols[0]].hist()
    hist_path = os.path.join(OUTPUT_DIR, "histogram.png")
    plt.savefig(hist_path)
    plt.close()
    report.append(f"Histogram saved: {hist_path}")

    # Heatmap
    plt.figure(figsize=(6,4))
    sns.heatmap(df[numeric_cols].corr(), annot=True)
    heatmap_path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()
    report.append(f"Heatmap saved: {heatmap_path}")

    return "\n".join(report)


# -------------------------
# Agents
# -------------------------
cleaner_agent = AssistantAgent(
    name="data_cleaner",
    system_message="Explain the dataset cleaning steps.",
    llm_config=llm_config,
)

feature_agent = AssistantAgent(
    name="feature_engineer",
    system_message="Explain engineered features.",
    llm_config=llm_config,
)

visualizer_agent = AssistantAgent(
    name="visualizer",
    system_message="Explain what charts were generated and why.",
    llm_config=llm_config,
)

analyzer_agent = AssistantAgent(
    name="analyzer",
    system_message="Provide insights based on dataset and visuals.",
    llm_config=llm_config,
)

user_agent = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    code_execution_config={"use_docker": False},
)

group_chat = GroupChat(
    agents=[
        user_agent,
        cleaner_agent,
        feature_agent,
        visualizer_agent,
        analyzer_agent
    ],
    messages=[],
    max_round=5,
    speaker_selection_method="round_robin",
    allow_repeat_speaker=False,
)

group_chat_manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
)

# -------------------------
# CSV upload
# -------------------------
@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    global RAW_DATASET, CLEAN_DATASET, FEATURE_DATASET
    global CLEANING_REPORT, FEATURE_REPORT, VISUAL_REPORT

    RAW_DATASET = pd.read_csv(file.file)
    CLEAN_DATASET, CLEANING_REPORT = clean_dataset(RAW_DATASET)
    FEATURE_DATASET, FEATURE_REPORT = engineer_features(CLEAN_DATASET)
    VISUAL_REPORT = generate_visuals(FEATURE_DATASET)

    return {
        "rows": len(FEATURE_DATASET),
        "columns": list(FEATURE_DATASET.columns),
        "cleaning_report": CLEANING_REPORT,
        "feature_report": FEATURE_REPORT,
        "visual_report": VISUAL_REPORT,
    }


# -------------------------
# Analyze
# -------------------------
@app.post("/analyze")
def analyze_data(request: AnalyzeRequest):
    global FEATURE_DATASET, group_chat

    if FEATURE_DATASET is None:
        return {"error": "No CSV uploaded yet."}

    # 🔁 RESET CHAT STATE (CRITICAL)
    group_chat.messages = []

    # 🔍 Lightweight dataset summary (optimized)
    summary = FEATURE_DATASET.describe(
        include="all"
    ).round(2).to_string(max_rows=10)

    prompt = f"""
DATA CLEANING REPORT:
{CLEANING_REPORT}

FEATURE ENGINEERING REPORT:
{FEATURE_REPORT}

VISUALIZATION REPORT:
{VISUAL_REPORT}

DATASET SUMMARY (LIMITED):
{summary}

USER QUESTION:
{request.query}
"""

    # 🚀 Run RoundRobin multi-agent chat
    user_agent.initiate_chat(
        group_chat_manager,
        message=prompt,
    )

    # 🧠 Safely extract analyzer response
    final_response = None
    for msg in reversed(group_chat.messages):
        sender = msg.get("name") or msg.get("sender") or msg.get("role")
        if sender == "analyzer":
            final_response = msg.get("content")
            break

    if not final_response:
        return {
            "analysis_plan": "Analyzer did not return a response.",
            "report_file": None
        }

    # 📄 Generate PDF report (post-processing)
    pdf_path = generate_pdf_report(
        analysis_text=final_response,
        output_dir=OUTPUT_DIR
    )

    return {
        "analysis_plan": final_response,
        "report_file": pdf_path,
        "charts_generated": os.listdir(OUTPUT_DIR)
    }
