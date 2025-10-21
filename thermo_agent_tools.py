import os
import json
import pandas as pd
from typing import Dict, Any, List, Optional
from langchain.prompts import PromptTemplate
import json
import re
import json5
import ast

def robust_json_parse(text: str) -> dict:
    """Tries multiple strategies to recover valid JSON from LLM output."""
    if hasattr(text, "content"):
        text = text.content

    # Strip Markdown formatting
    text = text.strip().removeprefix("```json").removesuffix("```").strip()

    # Try to extract first complete JSON object or array
    match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    if match:
        text = match.group(1)

    # Clean trailing commas
    text = re.sub(r',\s*([\]}])', r'\1', text)

    # Replace invalid constructs
    text = text.replace("None", "null")
    text = text.replace("'", '"')

    # Try standard JSON parse
    try:
        return json.loads(text)
    except:
        pass

    # Try JSON5
    try:
        return json5.loads(text)
    except:
        pass

    # Try Python literal eval (if it looks like a dict)
    try:
        return ast.literal_eval(text)
    except Exception as e:
        print("❌ JSON parse failed completely:", e)
        return {"materials": []}

# Fallback JSON parser for malformed strings
def parse_malformed_json(text):
    if hasattr(text, "content"):  # AIMessage from .invoke()
        text = text.content

    # Strip markdown ticks and whitespace
    text = text.strip().removeprefix("```json").removesuffix("```").strip()

    # Remove anything after the final closing brace/bracket
    # Capture either {...} or [...]
    json_match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    else:
        print("⚠️ No valid JSON-looking structure found.")
        return {"materials": []}

    # Clean common LLM mistakes
    text = re.sub(r",\s*([\]}])", r"\1", text)  # trailing commas
    text = text.replace("None", "null")         # Python → JSON
    text = text.replace("'", '"')               # single quotes

    try:
        return json.loads(text)
    except Exception as e:
        with open("llm_broken_output.txt", "w") as f:
            f.write(text)
        print("⚠️ Final JSON parse failed:", e)
        return {"materials": []}


# === Tool 1: Read fulltext ===
def read_fulltext(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# === Tool 2: Fast material candidate finder (any thermo-property mentioned) ===
def extract_material_candidates(fulltext: str, llm, max_materials: int = 20) -> List[str]:
    """
    Returns a deduplicated list of material names that have ANY thermoelectric
    property mentioned nearby in the text (ZT, S, σ, ρ, PF, κ).
    This is a lightweight pre-filter used to seed material hints downstream.
    """
    prompt = PromptTemplate.from_template("""
You are a scientific reading assistant. From the text below, list material names
(compounds, alloys, doped variants like "Bi2Te3", "SnSe:Na", "PbTe-AgSbTe2", "TiS2", "PEDOT:PSS", etc.)
that have ANY thermoelectric property mentioned close by (e.g., ZT, Seebeck S, electrical conductivity σ,
resistivity ρ, power factor PF, thermal conductivity κ). 

Rules:
- Only include materials for which at least one thermo property is discussed.
- Keep names as they appear (including dopants/phase labels when relevant).
- Return a JSON object with a single array "materials".
- Deduplicate.
- Limit to the first {max_materials} items.

Return JSON:
{{
  "materials": ["...", "..."]
}}

Text:
```{fulltext}```
""")
    out = llm.invoke(prompt.format(fulltext=fulltext, max_materials=max_materials))
    data = robust_json_parse(out.content)
    mats = data.get("materials", [])
    # Normalize to unique list of non-empty strings
    seen = set()
    result = []
    for m in mats:
        if not m or not isinstance(m, str):
            continue
        key = m.strip()
        if key and key not in seen:
            seen.add(key)
            result.append(key)
    return result


# === Tool 3: Thermoelectric Properties (now accepts material_names hint) ===
def extract_thermo_properties(fulltext: str, llm, material_names: Optional[List[str]] = None) -> Dict:
    material_hint = ""
    if material_names:
        formatted = ", ".join(f'"{name}"' for name in material_names)
        material_hint = (
            "Only extract entries for the following materials (ignore others unless name clearly matches variants): "
            f"{formatted}.\n"
        )

    prompt = PromptTemplate.from_template("""
You are a research extraction agent for thermoelectric materials.
{material_hint}
Extract per-material properties:
- name only, nothing extra string labels.
- ZT (figure of merit)
- σ (electrical conductivity)
- S (Seebeck coefficient)
- PF (power factor)
- κ (thermal conductivity)
- ρ (electrical resistivity)

For each property, extract **numeric values** along with **temperature** and **units** if mentioned.

Instructions:
- Set missing values to null strictly.
- Only include materials name nothing extra string labels.
- Don't do any calculation or unit conversion on your own.
- If multiple values exist, return all of them as separate dictionary entries.
- If more than 10 materials are found, include only the first 10.
- All field names and string values must use **valid JSON syntax** (double quotes).
- Keep numerical values unquoted (i.e., not strings).
- Nothing else should be included in the output strictly.

Return structured JSON:
{{
  "materials": [
    {{
      "name": "...",
      "zt_values": [{{"value": ..., "ZT_temperature": ..., "ZT_temperature_unit": "..."}}],
      "electrical_conductivity": [{{"σ_value": ..., "σ_unit": "...", "σ_Temperature": "...", "σ_Temp_unit": "..."}}],
      "electrical_resistivity": [{{"ρ_value": ..., "ρ_unit": "...", "ρ_Temperature": "...", "ρ_Temp_unit": "..."}}],                                   
      "seebeck_coefficient": [{{"S_value": ..., "S_unit": "...",  "S_Temperature": "...", "S_Temp_unit": "..."}}],
      "power_factor": [{{"PF_value": ..., "PF_unit": "...", "PF_Temperature": "...", "PF_Temp_unit": "..."}}],
      "thermal_conductivity": [{{"κ_value": ..., "κ_unit": "...", "κ_Temperature": "...", "κ_Temp_unit": "..."}}]
    }}
  ]
}}

Text:
```{fulltext}```
""")
    output = llm.invoke(prompt.format(fulltext=fulltext, material_hint=material_hint))
    return robust_json_parse(output.content)


# === Tool 4: Structural Properties (already supported: material_names hint) ===
def extract_structural_properties(fulltext: str, llm, material_names: list = None) -> Dict:
    material_hint = ""
    if material_names:
        formatted = ", ".join(f'"{name}"' for name in material_names)
        material_hint = f"Only extract structural properties for the following materials: {formatted}.\n"
    prompt = PromptTemplate.from_template("""
You are a structural extraction agent for thermoelectric materials.
{material_hint}
For each material, extract:
- name only, nothing extra string labels.
- compound_type, crystal_structure, lattice_structure
- space_group
- doping_type and list of dopants
- processing_method
Instructions:
- Set missing values to null strictly.
- Only include materials name nothing extra string labels.
- If more than 10 materials are found, include only the first 10.
- All field names and string values must use **valid JSON syntax** (double quotes).
- Set missing values to null strictly.
- All field values must follow **valid JSON syntax** with double quotes.
- Nothing else should be included in the output strictly.
                                          
Return JSON:
{{
  "materials": [
    {{
      "name": "...",
      "compound_type": "<type|null>",
      "crystal_structure": "<structure|null>",
      "lattice_structure": "<structure|null>",
      "space_group": "<group|null>",
      "doping": {{
        "doping_type": "<type|null>",
        "dopants": [<strings>]
      }},
      "processing_method": "<string|null>"
    }}
  ]
}}

Text:
```{fulltext}```
""")
    output = llm.invoke(prompt.format(fulltext=fulltext, material_hint=material_hint))
    return robust_json_parse(output.content)


# === Tool 5: Table extractor (already supports material_names) ===
def extract_from_tables(table_data: list, llm, material_names: list = None) -> dict:
    """Combine all tables and captions, and extract both thermo and structural fields."""
    if not table_data:
        return {"materials": []}

    # Material hint
    material_hint = ""
    if material_names:
        formatted = ", ".join(f'"{name}"' for name in material_names)
        material_hint = f"Only extract materials among the following: {formatted}.\n"

    # Combine all tables into one long string
    combined_block = ""
    for i, table in enumerate(table_data, 1):
        combined_block += f"### Table {i} Caption:\n{table['caption']}\n\n"
        combined_block += f"### Table {i} CSV Data:\n{json.dumps(table['rows'], indent=2)}\n\n"

    # Prompt
    prompt = f"""
You are a scientific table extraction agent working on thermoelectric materials.
{material_hint}
Below is a collection of tables and their captions from a scientific paper.

Extract all materials mentioned across the tables and return the following properties for each:

Thermoelectric:
- ZT (figure of merit)
- Seebeck coefficient (S)
- Electrical conductivity (σ)
- Electrical resistivity (ρ)
- Power factor (PF)
- Thermal conductivity (κ)

Structural:
- compound_type
- crystal_structure
- lattice_structure
- space_group
- doping_type and list of dopants
- processing_method

Instructions:
- Set missing values to null strictly.
- Only include materials name nothing extra string labels.
- If more than 10 materials are found, include only the first 10.
- All field names and string values must use **valid JSON syntax** (double quotes).
- Set missing values to null strictly.
- All field values must follow **valid JSON syntax** with double quotes.
- Nothing else should be included in the output strictly.

Return structured JSON like:
{{
  "materials": [
    {{
      "name": "...",
      "zt_values": [{{"value": ..., "ZT_temperature": ..., "ZT_temperature_unit": "..."}}],
      "electrical_conductivity": [{{"σ_value": ..., "σ_unit": "...", "σ_Temperature": "...", "σ_Temp_unit": "..."}}],
      "electrical_resistivity": [{{"ρ_value": ..., "ρ_unit": "...", "ρ_Temperature": "...", "ρ_Temp_unit": "..."}}],
      "seebeck_coefficient": [{{"S_value": ..., "S_unit": "...",  "S_Temperature": "...", "S_Temp_unit": "..."}}],
      "power_factor": [{{"PF_value": ..., "PF_unit": "...", "PF_Temperature": "...", "PF_Temp_unit": "..."}}],
      "thermal_conductivity": [{{"κ_value": ..., "κ_unit": "...", "κ_Temperature": "...", "κ_Temp_unit": "..."}}],
      "compound_type": "<type|null>",
      "crystal_structure": "<structure|null>",
      "lattice_structure": "<structure|null>",
      "space_group": "<group|null>",
      "doping": {{
        "doping_type": "<type|null>",
        "dopants": [<strings>]
      }},
      "processing_method": "<string|null>"
    }}
  ]
}}

All missing values must be explicitly set as null strictly.

### Tables and Captions:
{combined_block}
"""
    try:
        output = llm.invoke(prompt)
        return robust_json_parse(output.content)
    except Exception as e:
        print("❌ Table extraction failed:", e)
        return {"materials": []}

# === Tool 6: LLM-as-Judge Validator (thermoelectric numeric + structural check) ===
def judge_verify_properties(fulltext: str,
                            thermo_json: dict = None,
                            structure_json: dict = None,
                            table_json: dict = None,
                            llm=None,
                            folder_name: str = None) -> dict:
    """
    Validates thermoelectric numeric values (ZT, S, σ, ρ, PF, κ) and their temperature context
    against the full text and table data.  Also confirms each material has valid structural fields.
    Keeps all materials; removes only numeric values judged incorrect or temperature-mismatched.
    Logs [ok], [removed], [temp-mismatch], and [structure_ok] to judge_validation_log.txt.
    """
    from langchain.prompts import PromptTemplate
    import json, datetime, os
    from thermo_agent_tools import robust_json_parse

    # --- Merge all extracted JSON blocks ---
    merged = {"materials": []}
    for block in (thermo_json, structure_json, table_json):
        if block and block.get("materials"):
            merged["materials"].extend(block["materials"])

    if not merged["materials"]:
        return {"materials": [], "deleted": [], "notes": "No materials to judge."}

    # --- Build table context (captions + rows) if available ---
    table_context = ""
    try:
        if table_json and "table_data" in table_json:
            all_tables = table_json["table_data"]
        else:
            all_tables = []
    except Exception:
        all_tables = []

    if all_tables:
        table_context += "\n\n### Table Contexts (from paper):\n"
        for i, t in enumerate(all_tables, 1):
            caption = t.get("caption", "")
            rows = json.dumps(t.get("rows", []), indent=2)
            table_context += f"\nTable {i} Caption:\n{caption}\n\nRows:\n{rows}\n"

    # --- Prompt: thermoelectric numeric + temperature + structural verification ---
    prompt = PromptTemplate.from_template("""
You are a scientific verifier.

Given the full paper text, any tables with captions, and the extracted materials JSON,
validate each **thermoelectric numeric property** and its **temperature context**.

Thermoelectric properties to check only:
- ZT (figure of merit)
- Seebeck coefficient (S)
- Electrical conductivity (σ)
- Electrical resistivity (ρ)
- Power factor (PF)
- Thermal conductivity (κ)

Rules for numeric validation:
- For each numeric value, check whether the same or close value (and its temperature, if mentioned)
  appears in the text or tables.
- If both value and temperature are consistent → mark as correct.
- If value is correct but the temperature is inconsistent → mark as temp_mismatch.
- If value not found or implausible → mark as incorrect.
- If temperature is missing but value matches → mark as correct_no_temp.
- Never invent, average, or modify numbers.

Rules for structural validation:
- For each material, check whether its structural fields (compound_type, crystal_structure,
  lattice_structure, space_group, doping_type, processing_method) exist and look syntactically valid.
- If all these fields exist or are null (but properly formatted) → mark structure_ok.
- Do not delete structural information.

Return strictly valid JSON:
{{
  "correct": {{
     "Material": {{"ZT": [1.2], "S": [220]}},
  }},
  "incorrect": {{
     "Material": {{"ZT": [3.5]}},
  }},
  "temp_mismatch": {{
     "Material": {{"ZT": [{{"value":1.2,"reported_T":300,"found_T":700}}]}},
  }},
  "structure_ok": ["MaterialA","MaterialB"],
  "notes": "short explanation"
}}

Always include all four top-level keys even if empty.
Use double quotes for all keys and string values.
Enclose entire output in {{ }}.

### Full Paper Text
```{fulltext}```

### Table Captions and Data
```{table_context}```

### Merged JSON of Extracted Values
```{merged_json}```
""")

    # --- Run model ---
    res = llm.invoke(prompt.format(
        fulltext=fulltext,
        table_context=table_context,
        merged_json=json.dumps(merged, indent=2)
    ))

    # --- Parse output safely ---
    try:
        verdict = robust_json_parse(res.content)
        if not isinstance(verdict, dict):
            raise ValueError("Judge output is not a dict")

        correct = verdict.get("correct", {}) or {}
        incorrect = verdict.get("incorrect", {}) or {}
        temp_mismatch = verdict.get("temp_mismatch", {}) or {}
        structure_ok = verdict.get("structure_ok", []) or []
        notes = verdict.get("notes", "")
    except Exception as e:
        with open("judge_error_log.txt", "a", encoding="utf-8") as log:
            log.write("\n" + "=" * 60 + "\n")
            log.write(f"TIME: {datetime.datetime.now().isoformat()}\n")
            log.write(f"FOLDER: {folder_name or os.path(os.getcwd())}\n")
            log.write(f"ERROR: {repr(e)}\n")
            log.write(f"RAW OUTPUT:\n{res.content if hasattr(res, 'content') else str(res)}\n")
            log.write("=" * 60 + "\n")
        print(f"⚠️ Judge parsing failed → {e}")
        return merged  # fallback: keep everything

    # --- Local filtering + validation logging ---
    cleaned = []
    log_lines = []

    for mat in merged["materials"]:
        name = mat.get("name", "")

        mat_incorrect = incorrect.get(name, {})
        mat_temp_mismatch = temp_mismatch.get(name, {})
        mat_correct = correct.get(name, {})

        # Remove incorrect numeric values
        for key, bad_values in mat_incorrect.items():
            for prop_key in list(mat.keys()):
                if key.lower() in prop_key.lower() and isinstance(mat[prop_key], list):
                    before = len(mat[prop_key])
                    mat[prop_key] = [
                        v for v in mat[prop_key]
                        if not any(
                            (isinstance(bad, (int, float)) and bad == v.get("value"))
                            or (isinstance(bad, (int, float)) and bad == v.get(f"{key}_value"))
                            for bad in bad_values
                        )
                    ]
                    after = len(mat[prop_key])
                    if before > after:
                        for val in bad_values:
                            log_lines.append(f"[removed] {name}.{key}={val}")

        # Remove temperature-mismatched values
        for key, mismatch_list in mat_temp_mismatch.items():
            for prop_key in list(mat.keys()):
                if key.lower() in prop_key.lower() and isinstance(mat[prop_key], list):
                    for bad in mismatch_list:
                        val = bad.get("value")
                        mat[prop_key] = [
                            v for v in mat[prop_key]
                            if val != v.get("value") and val != v.get(f"{key}_value")
                        ]
                        log_lines.append(f"[temp-mismatch] {name}.{key} value={val} "
                                         f"reported_T={bad.get('reported_T')} found_T={bad.get('found_T')}")

        # Log correct and structural info
        for key, ok_values in mat_correct.items():
            for val in ok_values:
                log_lines.append(f"[ok] {name}.{key}={val}")

        if name in structure_ok:
            log_lines.append(f"[structure_ok] {name}")

        cleaned.append(mat)

    # --- Write validation log ---
    with open("judge_validation_log.txt", "a", encoding="utf-8") as log:
        log.write("\n" + "=" * 60 + "\n")
        log.write(f"TIME: {datetime.datetime.now().isoformat()}\n")
        log.write(f"FOLDER: {folder_name or os.path.basename(os.getcwd())}\n")
        log.write("\n".join(log_lines))
        log.write("\n" + "=" * 60 + "\n")

    print(f"🧾 Judge log: {len(log_lines)} entries validated for this folder.")
    return {"materials": cleaned, "notes": notes}
