# === langgraph_thermo_pipeline.py ===
import os
import json
import re
import ast
import json5
import random
import time
import pandas as pd
from pathlib import Path
from typing import Dict, Any, TypedDict, Optional
from typing_extensions import Annotated

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END
from thermo_agent_tools import (
    extract_material_candidates,
    extract_thermo_properties,
    extract_structural_properties,
    read_fulltext,
    extract_from_tables
)

# === LangGraph State ===
class State(TypedDict):
    folder: Path
    fulltext: Optional[str]
    llm: Optional[Any]
    material_names: Optional[list]      # ← now filled by candidate finder first
    thermo: Optional[dict]
    structure: Optional[dict]
    retries: int
    table_data: Optional[list]
    table_json_output: Optional[dict]
    total_table_rows: Optional[int]
    skip: bool

# === Node 1: Read Fulltext ===
def read_node(state: State) -> State:
    text = read_fulltext(state["folder"] / "fulltext.txt")
    return {**state, "fulltext": text, "retries": 0}

# === Node 2: Set dynamic max_tokens from token_count.txt ===
def set_tokens_node(state: State) -> State:
    folder = state["folder"]
    token_file = folder / "token_count.txt"

    try:
        with open(token_file, "r") as f:
            token_count = int(f.read().strip())
    except Exception:
        print(f"⚠️ Could not read token_count.txt in {folder.name}, defaulting to 999.")
        token_count = 999

    if token_count == 0:
        print(f"⏭️ Skipping {folder.name} due to token_count = 0")
        return {**state, "skip": True}  # flag to trigger early end

    # Compute max_tokens
    if token_count <= 1000:
        max_tok = 786
    elif token_count <= 3000:
        max_tok = 1024
    else:
        extra = (token_count - 3000) // 1000
        max_tok = 1024 + (512 * (extra + 1))
        max_tok = min(max_tok, 5120)

    print(f"🧠 Setting max_tokens = {max_tok} for {folder.name} (token_count = {token_count})")

    dynamic_llm = AzureChatOpenAI(
        azure_deployment="gpt-4.1",
        azure_endpoint="Endpoint_here",
        api_key="API_key_here",
        api_version="2025-01-01-preview",
        temperature=0.001,
        max_tokens=max_tok
    )
    return {**state, "llm": dynamic_llm, "skip": False}

# === Node 3: Find materials that have ANY thermo property ===
def find_materials_node(state: State) -> State:
    # Use a small fixed-token LLM just for this step
    small_llm = AzureChatOpenAI(
        azure_deployment="gpt-4.1-mini",
        azure_endpoint="Endpoint_here",
        api_key="API_key_here",
        api_version="2025-01-01-preview",
        temperature=0.001,
        max_tokens=256   # fixed small cap for efficiency
    )
    candidates = extract_material_candidates(state["fulltext"], llm=small_llm, max_materials=20)
    if candidates:
        print(f"🧪 Candidate materials (thermo-mentioned): {len(candidates)} → {candidates[:8]}{'...' if len(candidates) > 8 else ''}")
        return {**state, "material_names": candidates, "skip": False}
    else:
        print("🛑 No thermo-related materials found → skipping downstream extraction.")
        return {**state, "material_names": [], "skip": True}



# === Node 4: Extract Thermoelectric Properties (uses hint if available) ===
def extract_thermo_node(state: State) -> State:
    thermo = extract_thermo_properties(
        state["fulltext"],
        llm=state["llm"],
        material_names=state.get("material_names") or None
    )
    if not thermo.get("materials") or not isinstance(thermo["materials"], list):
        raise ValueError("❌ Thermo extraction returned no valid materials.")
    return {**state, "thermo": thermo}

# === Node 5: Extract Structural Properties (uses same hint) ===
def extract_structure_node(state: State) -> State:
    # Prefer the candidate list; if thermo produced better names, you can also merge.
    hint_names = state.get("material_names") or []
    # Optional improvement: union with names discovered by thermo to tighten recall.
    thermo_names = [m.get("name") for m in state.get("thermo", {}).get("materials", []) if m.get("name")]
    merged = list(dict.fromkeys([*hint_names, *thermo_names]))  # preserve order & dedupe
    struct = extract_structural_properties(state["fulltext"], llm=state["llm"], material_names=merged or None)
    if not struct.get("materials"):
        raise ValueError("❌ Structure JSON parse failed or empty.")
    return {**state, "structure": struct, "material_names": merged}

# === Node 6: table rows_count ===
def count_table_and_plan_tokens_node(state: State) -> State:
    folder = state["folder"]
    table_data = []
    total_rows = 0
    i = 1

    while True:
        csv_path = folder / f"table{i}.csv"
        caption_path = folder / f"table{i}_caption.txt"
        if not csv_path.exists() or not caption_path.exists():
            break
        try:
            df = pd.read_csv(csv_path)
            with open(caption_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()
            row_count = len(df)
            table_data.append({
                "filename": f"table{i}.csv",
                "caption": caption,
                "rows": df.to_dict(orient="records"),
                "row_count": row_count
            })
            total_rows += row_count
        except Exception as e:
            print(f"⚠️ Failed reading {csv_path.name}: {e}")
        i += 1

    # Initialize max_tokens based on row count
    if total_rows == 0:
        max_tokens = 512  # Default to 512 if no rows are found
    else:
        max_tokens = min(512 + total_rows * 325, 5120)  # Heuristic calculation

    # Ensure dynamic_llm always gets updated with max_tokens
    dynamic_llm = AzureChatOpenAI(
        azure_deployment="gpt-4.1-mini",
        azure_endpoint="Endpoint_here",
        api_key="API_key_here",
        api_version="2025-01-01-preview",
        temperature=0.001,
        max_tokens=max_tokens
    )

    print(f"📊 Found {len(table_data)} tables with {total_rows} rows → max_tokens = {max_tokens}")

    return {
        **state,
        "table_data": table_data,
        "total_table_rows": total_rows,
        "llm": dynamic_llm  # override with expanded LLM
    }

# === Node 7: Extract from Tables (uses same material hint) ===
def extract_table_json_node(state: State) -> State:
    if not state.get("table_data"):
        return {**state, "table_json_output": {"materials": []}}

    output = extract_from_tables(
        table_data=state["table_data"],
        llm=state["llm"],
        material_names=state.get("material_names")
    )
    return {**state, "table_json_output": output}

# === Node 8: Save Output ===
def write_node(state: State) -> State:
    folder = state["folder"]
    with open(folder / "t.json", "w") as f:
        json.dump(state["thermo"], f, indent=2)
    with open(folder / "s.json", "w") as f:
        json.dump(state["structure"], f, indent=2)
    if state.get("table_json_output") and state["table_json_output"].get("materials"):
        with open(folder / "tables_output.json", "w") as f:
            json.dump(state["table_json_output"], f, indent=2)
    print(f"✅ Done: {folder.name}")
    return state

# === Retry Decision Function ===
def skip_if_zero_tokens(state: State) -> str:
    return END if state.get("skip") else "Find_materials"

def skip_if_no_materials(state: State) -> str:
    return END if state.get("skip") else "Thermoelectric_prop"

def table_branch(state: State) -> str:
    return "has_tables" if state.get("table_data") else "no_tables"

# === Graph Wiring ===
graph = StateGraph(State)
graph.add_node("read_file", read_node)
graph.add_node("set_tokens", set_tokens_node)
graph.add_node("Find_materials", find_materials_node)        
graph.add_node("Thermoelectric_prop", extract_thermo_node)
graph.add_node("Structural_prop", extract_structure_node)
graph.add_node("Plan_table_tokens", count_table_and_plan_tokens_node)
graph.add_node("Extract_table_JSON", extract_table_json_node)
graph.add_node("Write_json", write_node)

graph.set_entry_point("read_file")
graph.add_edge("read_file", "set_tokens")

graph.add_conditional_edges("set_tokens", skip_if_zero_tokens, {
    END: END,
    "Find_materials": "Find_materials"
})

graph.add_conditional_edges("Find_materials", skip_if_no_materials, {
    END: END,
    "Thermoelectric_prop": "Thermoelectric_prop"
})

graph.add_edge("Thermoelectric_prop", "Structural_prop")
graph.add_edge("Structural_prop", "Plan_table_tokens")



graph.add_conditional_edges("Plan_table_tokens", table_branch, {
    "has_tables": "Extract_table_JSON",
    "no_tables": "Write_json"
})

graph.add_edge("Extract_table_JSON", "Write_json")
graph.add_edge("Write_json", END)

# === Compile ===
app = graph.compile()
print("📊 Graph compiled.")

# === Run for next N folders not already completed ===
base_dir = Path("elsevier_gpt_processed_articles")
completed_log = Path("completed_folders_gpt.txt")
max_new_folders = 2000
new_count = 0

# === Load completed and failed folder names ===
completed_folders = set()
failed_folders = set()
if completed_log.exists():
    with open(completed_log, "r") as f:
        completed_folders = set(line.strip() for line in f)

failed_log = Path("failed_folders_gpt.txt")
if failed_log.exists():
    with open(failed_log, "r") as f:
        failed_folders = set(line.strip() for line in f)

# === Process folders ===
for folder in base_dir.iterdir():
    if not folder.is_dir():
        continue
    if folder.name in completed_folders or folder.name in failed_folders:
        continue  # Already processed or failed

    try:
        print(f"🚀 Running on folder {new_count + 1}: {folder.name}")
        app.invoke(State(
            folder=folder,
            fulltext=None,
            llm=None,
            material_names=None,
            thermo=None,
            structure=None,
            retries=0,
            skip=False,
            table_data=None,
            table_json_output=None,
            total_table_rows=0
        ))
        # ✅ Log success
        with open(completed_log, "a") as log_file:
            log_file.write(f"{folder.name}\n")
        new_count += 1

    except Exception as e:
        print(f"⚠️ Failed on {folder.name}: {e}")
        # ❌ Log failure
        with open(failed_log, "a") as f:
            f.write(f"{folder.name}\n")

    # Throttle
    t = random.uniform(6, 10)
    print(f"⏳ Sleeping for {t:.2f} seconds before next folder...")
    time.sleep(t)

    if new_count >= max_new_folders:
        print(f"🔚 Reached limit of {max_new_folders} new folders.")
        break
