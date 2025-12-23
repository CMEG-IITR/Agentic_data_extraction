# Agentic Data Extraction for Thermoelectric Materials

An intelligent, multi-agent system for automated extraction of thermoelectric and structural properties from scientific papers using LLM-powered agents and LangGraph workflows.

## 🔬 Overview

This project implements an agentic data extraction pipeline specifically designed for thermoelectric materials research. It processes scientific papers to automatically extract:

- **Thermoelectric Properties**: ZT (figure of merit), Seebeck coefficient (S), electrical conductivity (σ), electrical resistivity (ρ), power factor (PF), and thermal conductivity (κ)
- **Structural Properties**: compound type, crystal structure, lattice structure, space group, doping information, and processing methods
- **Tabular Data**: Extracts and validates data from tables within papers

The system uses a graph-based workflow orchestrated by LangGraph, with multiple specialized agent nodes that collaborate to ensure high-quality extraction and validation.

## ✨ Key Features

### 🤖 Multi-Agent Architecture
- **Material Candidate Finder**: Identifies materials with thermoelectric properties mentioned
- **Thermo Extractor**: Extracts thermoelectric properties with temperature context
- **Structure Extractor**: Captures structural and compositional information
- **Table Processor**: Parses and extracts data from paper tables
- **LLM Judge**: Validates extracted data against source text for accuracy

### 🧠 Intelligent Processing
- **Dynamic Token Allocation**: Automatically adjusts LLM token limits based on document size
- **Smart Skip Logic**: Bypasses papers without relevant content to save resources
- **Temperature-Aware Validation**: Verifies that property values match reported temperatures
- **Robust JSON Parsing**: Multiple fallback strategies for handling malformed LLM outputs

### 📊 Data Quality
- **Cross-Validation**: Judge agent verifies extracted values against original text
- **Duplicate Detection**: Deduplicates materials and properties
- **Error Logging**: Comprehensive logging of failures and validation issues
- **Retry Mechanisms**: Handles transient failures gracefully

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Azure OpenAI API key or Google Generative AI API key

### Dependencies

Install the required packages:

```bash
pip install langchain langchain-google-genai langchain-openai langgraph pandas json5
```

**Core Dependencies:**
- `langchain` - Framework for LLM applications
- `langchain-google-genai` - Google Gemini integration
- `langchain-openai` - Azure OpenAI integration
- `langgraph` - Graph-based workflow orchestration
- `pandas` - Data manipulation and CSV processing
- `json5` - Lenient JSON parsing
- `typing-extensions` - Advanced type hints

## 📋 Configuration

Before running the extraction pipeline, configure your API credentials in `run_thermo_agent.py`:

```python
# Azure OpenAI Configuration
model_name = "gpt-4.1-mini"
endpoint = "your-azure-endpoint"
api_key = "your-api-key"
api_version = "2024-12-01-preview"
```

## 🎯 Usage

### Basic Workflow

1. **Prepare Your Data**: Organize papers in folders with the following structure:
```
elsevier_gpt_processed_articles/
├── paper_folder_1/
│   ├── fulltext.txt
│   ├── token_count.txt
│   ├── table1.csv
│   ├── table1_caption.txt
│   └── ...
├── paper_folder_2/
│   └── ...
```

2. **Run the Extraction Pipeline**:
```bash
python run_thermo_agent.py
```

3. **Output Files**: For each processed paper, the system generates:
   - `t.json` - Thermoelectric properties
   - `s.json` - Structural properties
   - `tables_output.json` - Table-extracted data (if tables exist)

### Advanced Usage

#### Single Paper Processing
Modify the `run_thermo_agent.py` script to process a single folder:

```python
folder = Path("path/to/paper_folder")
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
```

#### Batch Processing with Rate Limiting
The default configuration processes up to 2000 papers with:
- Random delays (6-10 seconds) between papers
- 60-second cooldown every 10 papers to avoid rate limits

## 🏗️ Project Structure

```
.
├── run_thermo_agent.py        # Main orchestration script with LangGraph workflow
├── thermo_agent_tools.py      # Core extraction functions and tools
├── test.py                    # Test utilities
├── Inference.ipynb            # Inference demonstration notebook
├── data_preprocessing.ipynb   # Data preparation notebook
├── Post_processing_EDA.ipynb  # Results analysis notebook
├── judge_validation_log.txt   # Validation logs from judge agent
└── LICENSE                    # MIT License
```

### Core Components

#### `run_thermo_agent.py`
Implements the LangGraph state machine with these nodes:
1. **read_file** - Loads paper fulltext
2. **set_tokens** - Configures dynamic token limits
3. **Find_materials** - Identifies candidate materials
4. **Thermoelectric_prop** - Extracts thermoelectric properties
5. **Structural_prop** - Extracts structural information
6. **Plan_table_tokens** - Counts table rows and adjusts tokens
7. **Extract_table_JSON** - Processes tables
8. **Judge_verification** - Validates extracted data
9. **Write_json** - Saves results to files

#### `thermo_agent_tools.py`
Contains specialized extraction functions:
- `extract_material_candidates()` - Fast material identification
- `extract_thermo_properties()` - Thermoelectric property extraction
- `extract_structural_properties()` - Structural data extraction
- `extract_from_tables()` - Table parsing and extraction
- `judge_verify_properties()` - LLM-as-judge validation
- `robust_json_parse()` - Resilient JSON parsing with multiple fallbacks

## 📊 Data Flow

```
Paper (fulltext + tables)
    ↓
[Material Candidate Finder]
    ↓
[Parallel Extraction]
    ├─→ [Thermo Properties]
    └─→ [Structural Properties]
    ↓
[Table Extraction]
    ↓
[Judge Validation]
    ↓
Output JSON Files
```

## 🔧 Customization

### Adjusting Extraction Limits
Modify material limits in `run_thermo_agent.py`:
```python
candidates = extract_material_candidates(
    state["fulltext"], 
    llm=small_llm, 
    max_materials=20  # Adjust this value
)
```

### Changing LLM Models
Update model configuration:
```python
dynamic_llm = AzureChatOpenAI(
    azure_deployment="your-model-name",
    temperature=0.001,  # Adjust for consistency vs creativity
    max_tokens=max_tok
)
```

### Custom Property Extraction
Extend prompts in `thermo_agent_tools.py` to extract additional properties.

## 📝 Logging and Monitoring

The system generates several log files:

- **`judge_validation_log.txt`** - Detailed validation results with [ok], [removed], [temp-mismatch], and [structure_ok] tags
- **`judge_error_log.txt`** - Judge agent failures and parsing errors
- **`completed_folders_gpt.txt`** - Successfully processed papers
- **`failed_folders_gpt.txt`** - Papers that failed processing
- **`llm_broken_output.txt`** - Raw LLM outputs that failed JSON parsing

## 🛠️ Troubleshooting

### Common Issues

**JSON Parsing Errors**: The system includes robust fallback parsing with `json5` and `ast.literal_eval`. Check `llm_broken_output.txt` for problematic outputs.

**Rate Limiting**: Adjust sleep times in the main loop:
```python
t = random.uniform(6, 10)  # Increase these values
time.sleep(t)
```

**Token Limit Exceeded**: The system auto-adjusts, but you can modify the calculation in `set_tokens_node()` or `count_table_and_plan_tokens_node()`.

**Missing Dependencies**: Ensure all packages are installed:
```bash
pip install langchain langchain-google-genai langchain-openai langgraph pandas json5
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional property extractors
- Support for more document formats
- Enhanced validation logic
- Performance optimizations
- Better error recovery

Please ensure any contributions maintain the existing code style and include appropriate documentation.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎓 Citation

If you use this work in your research, please cite:

```
CMEG-IITR. (2025). Agentic Data Extraction for Thermoelectric Materials. 
GitHub repository: https://github.com/CMEG-IITR/Agentic_data_extraction
```

## 🔗 Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Azure OpenAI Service](https://azure.microsoft.com/en-us/products/ai-services/openai-service)

## 👥 Team

Developed by CMEG-IITR (Computational Materials Engineering Group, IIT Roorkee)

---

**Note**: This is a research tool designed for academic use. Always verify extracted data against original sources for critical applications.
