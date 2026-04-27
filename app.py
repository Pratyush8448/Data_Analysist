# -------------------------
# Standard Library
# -------------------------
import os
import re
import sys
import json
import time
import base64
import socket
import shutil
import tempfile
import subprocess
import logging
import traceback
import asyncio
from io import BytesIO
from datetime import datetime, timedelta
from functools import partial
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

# -------------------------
# Third-Party Libraries
# -------------------------
import requests
import httpx
import psutil
import networkx as nx

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import importlib.metadata

# -------------------------
# FastAPI
# -------------------------
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Query
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, Response

# -------------------------
# Environment
# -------------------------
from dotenv import load_dotenv

# Optional image conversion
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# LangChain / LLM imports (keep as you used)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TDS Data Analyst Agent")

# -------------------- LLM Multi-Provider Config --------------------
from collections import defaultdict
import time
import requests

# Load Gemini keys (clean + deduplicated)
GEMINI_KEYS = []
for i in range(1, 11):
    key = os.getenv(f"gemini_api_{i}")
    if key and key.strip():
        GEMINI_KEYS.append(key.strip())

GEMINI_KEYS = list(dict.fromkeys(GEMINI_KEYS))  # remove duplicates

# Grok config
GROK_API_KEY = os.getenv("GROK_API_KEY")
USE_GROK = os.getenv("USE_GROK", "false").lower() == "true"

# Model hierarchy (Gemini)
MODEL_HIERARCHY = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite"
]

MAX_RETRIES_PER_KEY = 2
TIMEOUT = 30
QUOTA_KEYWORDS = ["quota", "exceeded", "rate limit", "403", "too many requests"]

# -------------------- Validation --------------------
if not USE_GROK and not GEMINI_KEYS:
    raise RuntimeError("No Gemini API keys found.")

if USE_GROK and not GROK_API_KEY:
    raise RuntimeError("USE_GROK=true but GROK_API_KEY missing.")

# -------------------- LLM wrapper --------------------
class LLMWithFallback:
    def __init__(self, keys=None, models=None, temperature=0):
        self.keys = keys or GEMINI_KEYS
        self.models = models or MODEL_HIERARCHY
        self.temperature = temperature

        self.current_llm = None
        self.current_key = None
        self.current_model = None

        self.slow_keys_log = defaultdict(list)
        self.failing_keys_log = defaultdict(int)

    def _create_llm(self, key, model):
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=self.temperature,
            google_api_key=key
        )

    def _get_llm_instance(self):
        if self.current_llm:
            return self.current_llm

        last_error = None

        for model in self.models:
            for key in self.keys:
                for attempt in range(MAX_RETRIES_PER_KEY):
                    try:
                        llm_instance = self._create_llm(key, model)

                        # Save working config
                        self.current_llm = llm_instance
                        self.current_key = key
                        self.current_model = model

                        logger.info(f"Using Gemini model={model} key={key[:6]}...")
                        return llm_instance

                    except Exception as e:
                        last_error = e
                        msg = str(e).lower()

                        if any(qk in msg for qk in QUOTA_KEYWORDS):
                            self.slow_keys_log[key].append(model)

                        self.failing_keys_log[key] += 1
                        time.sleep(0.5)

        raise RuntimeError(f"All Gemini models/keys failed. Last error: {last_error}")

    def bind_tools(self, tools):
        llm = self._get_llm_instance()
        return llm.bind_tools(tools)

    def invoke(self, prompt):
        try:
            llm = self._get_llm_instance()
            return llm.invoke(prompt)

        except Exception as e:
            logger.warning(f"LLM failed, resetting instance: {e}")

            self.current_llm = None

            llm = self._get_llm_instance()
            return llm.invoke(prompt)


LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", 240))


@app.get("/", response_class=HTMLResponse)
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, "index.html")

        with open(file_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())

    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Frontend not found</h1><p>Ensure index.html exists</p>",
            status_code=404
        )


def parse_keys_and_types(raw_questions: str):
    """
    Extracts structured keys and their expected types from input text.
    """

    pattern = r"-\s*`([^`]+)`\s*:\s*([a-zA-Z]+)"
    matches = re.findall(pattern, raw_questions)

    if not matches:
        return [], {}

    type_map_def = {
        "number": float,
        "float": float,
        "integer": int,
        "int": int,
        "string": str
    }

    keys_list = []
    type_map = {}

    for key, t in matches:
        t_clean = t.strip().lower()

        keys_list.append(key)
        type_map[key] = type_map_def.get(t_clean, str)

    return keys_list, type_map




@tool
def scrape_url_to_dataframe(url: str) -> Dict[str, Any]:
    """
    Robust URL scraper → returns structured dataframe-like output
    """
    logger.info(f"Scraping URL: {url}")

    try:
        from io import BytesIO, StringIO
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.google.com/",
        }

        # -------------------------
        # Retry logic
        # -------------------------
        last_error = None
        for _ in range(2):
            try:
                resp = requests.get(url, headers=headers, timeout=15)
                resp.raise_for_status()
                break
            except Exception as e:
                last_error = e
                time.sleep(1)
        else:
            return {"status": "error", "message": str(last_error)}

        ctype = resp.headers.get("Content-Type", "").lower()
        content = resp.content[:2_000_000]  # limit size (~2MB)

        df = None

        # -------------------------
        # CSV
        # -------------------------
        if "text/csv" in ctype or url.lower().endswith(".csv"):
            df = pd.read_csv(BytesIO(content))

        # -------------------------
        # Excel
        # -------------------------
        elif any(url.lower().endswith(ext) for ext in (".xls", ".xlsx")):
            df = pd.read_excel(BytesIO(content))

        # -------------------------
        # Parquet
        # -------------------------
        elif url.lower().endswith(".parquet"):
            df = pd.read_parquet(BytesIO(content))

        # -------------------------
        # JSON
        # -------------------------
        elif "application/json" in ctype or url.lower().endswith(".json"):
            try:
                data = resp.json()
                df = pd.json_normalize(data)
            except Exception:
                df = pd.DataFrame([{"text": resp.text[:5000]}])

        # -------------------------
        # HTML
        # -------------------------
        elif "text/html" in ctype:
            html_content = resp.text

            # Try table extraction
            try:
                tables = pd.read_html(StringIO(html_content))
                if tables:
                    df = tables[0]
            except Exception:
                pass

            # Fallback → plain text
            if df is None or df.empty:
                soup = BeautifulSoup(html_content, "html.parser")
                text = soup.get_text(separator="\n", strip=True)

                df = pd.DataFrame({
                    "text": [text[:5000]]  # truncate
                })

        # -------------------------
        # Fallback
        # -------------------------
        if df is None or df.empty:
            df = pd.DataFrame({
                "text": [resp.text[:5000]]
            })

        # -------------------------
        # Normalize columns
        # -------------------------
        df.columns = (
            df.columns
            .map(str)
            .str.replace(r'\[.*?\]', '', regex=True)
            .str.strip()
        )

        # Limit rows (prevent huge payloads)
        df = df.head(200)

        return {
            "status": "success",
            "data": df.to_dict(orient="records"),
            "columns": df.columns.tolist()
        }

    except Exception as e:
        logger.exception("Scraping failed")
        return {"status": "error", "message": str(e)}

def clean_llm_output(output: str) -> Dict:
    """
    Extract the most likely valid JSON object from LLM output.
    Returns parsed dict or {"error": "..."}
    """
    try:
        if not output:
            return {"error": "Empty LLM output"}

        s = output.strip()

        # Remove markdown fences
        s = re.sub(r"^```(?:json)?", "", s)
        s = re.sub(r"```$", "", s)

        # -------------------------
        # Extract all JSON candidates using brace stack
        # -------------------------
        stack = []
        start = -1
        candidates = []

        for i, ch in enumerate(s):
            if ch == "{":
                if not stack:
                    start = i
                stack.append(ch)

            elif ch == "}":
                if stack:
                    stack.pop()
                    if not stack and start != -1:
                        candidates.append(s[start:i+1])

        if not candidates:
            return {"error": "No JSON found", "raw": s}

        # -------------------------
        # Try parsing (largest first)
        # -------------------------
        for cand in sorted(candidates, key=len, reverse=True):
            try:
                return json.loads(cand)
            except Exception:
                continue

        return {
            "error": "JSON parsing failed",
            "raw": candidates[-1]
        }

    except Exception as e:
        return {"error": str(e)}
    

SCRAPE_FUNC = r'''
from typing import Dict, Any
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from io import StringIO

def scrape_url_to_dataframe(url: str) -> Dict[str, Any]:
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=8
        )
        response.raise_for_status()
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "data": [],
            "columns": []
        }

    content_type = response.headers.get("Content-Type", "").lower()
    html = response.text[:1_000_000]  # limit size

    df = None

    # -------------------------
    # Try extracting tables
    # -------------------------
    if "html" in content_type:
        try:
            tables = pd.read_html(StringIO(html))
            if tables:
                df = tables[0]
        except Exception:
            df = None

    # -------------------------
    # Fallback → text
    # -------------------------
    if df is None or df.empty:
        soup = BeautifulSoup(html, "html.parser")
        text_data = soup.get_text(separator="\n", strip=True)

        # simple fallback dataframe
        df = pd.DataFrame({
            "text": [text_data[:3000]]  # truncate
        })

    # -------------------------
    # Normalize columns
    # -------------------------
    df.columns = [
        str(col).strip() for col in df.columns
    ]

    # limit rows
    df = df.head(100)

    return {
        "status": "success",
        "data": df.to_dict(orient="records"),
        "columns": list(df.columns)
    }
'''


def write_and_run_temp_python(code: str, injected_pickle: str = None, timeout: int = 60) -> Dict[str, Any]:
    """
    Safe execution sandbox for LLM-generated code
    """

    preamble = [
        "import json, sys, gc",
        "import pandas as pd, numpy as np",
        "import matplotlib",
        "matplotlib.use('Agg')",
        "import matplotlib.pyplot as plt",
        "from io import BytesIO",
        "import base64",
    ]

    if PIL_AVAILABLE:
        preamble.append("from PIL import Image")

    if injected_pickle:
        preamble.append(f"df = pd.read_pickle(r'''{injected_pickle}''')")
        preamble.append("data = df.to_dict(orient='records')")
    else:
        preamble.append("data = globals().get('data', {})")

    helper = r'''
def plot_to_base64(max_bytes=100000):
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_bytes = buf.getvalue()

    if len(img_bytes) <= max_bytes:
        return base64.b64encode(img_bytes).decode('ascii')

    for dpi in [80, 60, 50, 40, 30]:
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
        buf.seek(0)
        b = buf.getvalue()
        if len(b) <= max_bytes:
            return base64.b64encode(b).decode('ascii')

    return base64.b64encode(img_bytes).decode('ascii')
'''

    script_lines = []
    script_lines.extend(preamble)
    script_lines.append(helper)
    script_lines.append(SCRAPE_FUNC)
    script_lines.append("\nresults = {}\n")

    # Basic safety filter
    if any(x in code for x in ["import os", "subprocess", "sys.exit", "open("]):
        return {"status": "error", "message": "Unsafe code detected"}

    script_lines.append(code)

    # Always ensure output
    script_lines.append("""
try:
    print(json.dumps({'status':'success','result':results}, default=str), flush=True)
except Exception as e:
    print(json.dumps({'status':'error','message':str(e)}), flush=True)
""")

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8')
    tmp.write("\n".join(script_lines))
    tmp.flush()
    tmp_path = tmp.name
    tmp.close()

    try:
        completed = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()

        if completed.returncode != 0:
            logger.error("Execution failed")
            logger.error("STDOUT:\n%s", stdout)
            logger.error("STDERR:\n%s", stderr)

            return {
                "status": "error",
                "message": stderr or stdout
            }

        # -------------------------
        # Extract JSON safely
        # -------------------------
        matches = re.findall(r"\{.*\}", stdout, re.DOTALL)

        if not matches:
            return {
                "status": "error",
                "message": "No JSON output",
                "raw": stdout
            }

        for cand in reversed(matches):
            try:
                return json.loads(cand)
            except Exception:
                continue

        return {
            "status": "error",
            "message": "JSON parsing failed",
            "raw": stdout
        }

    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Execution timed out"}

    finally:
        try:
            os.unlink(tmp_path)
            if injected_pickle and os.path.exists(injected_pickle):
                os.unlink(injected_pickle)
        except Exception:
            pass


# -----------------------------
# LLM Selection (UPDATED)
# -----------------------------
if USE_GROK:
    logger.info("Using Grok LLM")
    llm = GrokLLM(api_key=GROK_API_KEY)
else:
    logger.info(f"Using Gemini with {len(GEMINI_KEYS)} keys")
    llm = LLMWithFallback(temperature=0)


# -----------------------------
# Tools
# -----------------------------
tools = [scrape_url_to_dataframe]


# -----------------------------
# Prompt
# -----------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a full-stack autonomous data analyst agent.

You will receive:
- A set of **rules** for this request
- One or more **questions**
- An optional **dataset preview**

You must:
1. Follow the provided rules exactly.
2. Return only a valid JSON object — no extra commentary.
3. JSON must contain:
   - "questions": [...]
   - "code": "..." (Python code that fills `results` dict)

4. Execution environment:
   - pandas, numpy, matplotlib available
   - helper: plot_to_base64(max_bytes=100000)

5. Always define variables before use.
6. Code must run without errors.
"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


# -----------------------------
# Agent Creation
# -----------------------------
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)


# -----------------------------
# Agent Executor (Improved)
# -----------------------------
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,

    max_iterations=5,

    # Better stopping
    early_stopping_method="generate",

    # Handle bad outputs more gracefully
    handle_parsing_errors="Check your output and return valid JSON only.",

    # Keep clean output
    return_intermediate_steps=False
)


# -----------------------------
# Runner: orchestrates agent -> pre-scrape inject -> execute
def run_agent_safely(llm_input: str) -> Dict:
    """
    Orchestrates:
    1. Agent execution
    2. JSON extraction
    3. Optional scraping + dataframe injection
    4. Safe code execution
    """

    try:
        # -------------------------
        # Run agent
        # -------------------------
        response = agent_executor.invoke(
            {"input": llm_input},
            {"timeout": LLM_TIMEOUT_SECONDS}
        )

        raw_out = (
            response.get("output")
            or response.get("final_output")
            or response.get("text")
            or str(response)
        )

        if not raw_out:
            return {"error": f"No output from agent", "raw": response}

        logger.info("LLM RAW OUTPUT:\n%s", raw_out[:1000])

        # -------------------------
        # Parse JSON
        # -------------------------
        parsed = clean_llm_output(raw_out)

        if "error" in parsed:
            return parsed

        if not isinstance(parsed, dict):
            return {"error": "Parsed output is not dict", "raw": parsed}

        if "code" not in parsed or "questions" not in parsed:
            return {"error": "Missing required keys", "raw": parsed}

        code = parsed["code"]
        questions: List[str] = parsed["questions"]

        logger.info("EXECUTING CODE:\n%s", code[:1000])

        # -------------------------
        # Basic safety check
        # -------------------------
        if any(x in code for x in ["import os", "subprocess", "sys.exit", "open("]):
            return {"error": "Unsafe code detected"}

        # -------------------------
        # Detect and run scrape calls
        # -------------------------
        urls = list(set(re.findall(
            r"scrape_url_to_dataframe\(\s*['\"](.*?)['\"]\s*\)", code
        )))

        pickle_path = None

        if urls:
            # Use first URL (can extend later)
            url = urls[0]

            tool_resp = scrape_url_to_dataframe(url)

            if tool_resp.get("status") != "success":
                return {"error": f"Scrape failed: {tool_resp.get('message')}"}

            df = pd.DataFrame(tool_resp.get("data", []))

            if df.empty:
                return {"error": "Scraped dataframe is empty"}

            temp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
            temp_pkl.close()

            df.to_pickle(temp_pkl.name)
            pickle_path = temp_pkl.name

        # -------------------------
        # Execute code
        # -------------------------
        exec_result = write_and_run_temp_python(
            code,
            injected_pickle=pickle_path,
            timeout=LLM_TIMEOUT_SECONDS
        )

        if exec_result.get("status") != "success":
            return {
                "error": f"Execution failed",
                "message": exec_result.get("message"),
                "raw": exec_result.get("raw")
            }

        results_dict = exec_result.get("result", {})

        # -------------------------
        # Map results to questions
        # -------------------------
        output = {}

        for q in questions:
            output[q] = results_dict.get(q, "Answer not found")

        return output

    except Exception as e:
        logger.exception("run_agent_safely failed")
        return {"error": str(e)}



@app.post("/api")
async def analyze_data(request: Request):
    try:
        form = await request.form()

        questions_file = None
        data_file = None

        # -------------------------
        # Extract files
        # -------------------------
        for _, val in form.items():
            if hasattr(val, "filename") and val.filename:
                fname = val.filename.lower()
                if fname.endswith(".txt") and questions_file is None:
                    questions_file = val
                else:
                    data_file = val

        if not questions_file:
            raise HTTPException(400, "Missing questions file (.txt)")

        raw_questions = (await questions_file.read()).decode("utf-8")
        keys_list, type_map = parse_keys_and_types(raw_questions)

        pickle_path = None
        df_preview = ""
        dataset_uploaded = False

        # -------------------------
        # Handle dataset
        # -------------------------
        if data_file:
            dataset_uploaded = True
            filename = data_file.filename.lower()
            content = await data_file.read()
            from io import BytesIO

            try:
                if filename.endswith(".csv"):
                    df = pd.read_csv(BytesIO(content))

                elif filename.endswith((".xlsx", ".xls")):
                    df = pd.read_excel(BytesIO(content))

                elif filename.endswith(".parquet"):
                    df = pd.read_parquet(BytesIO(content))

                elif filename.endswith(".json"):
                    try:
                        df = pd.read_json(BytesIO(content))
                    except Exception:
                        df = pd.DataFrame(json.loads(content.decode("utf-8")))

                elif filename.endswith((".png", ".jpg", ".jpeg")):
                    if not PIL_AVAILABLE:
                        raise HTTPException(400, "PIL not available")

                    image = Image.open(BytesIO(content)).convert("RGB")
                    df = pd.DataFrame({"image": [image]})

                else:
                    raise HTTPException(400, f"Unsupported file type: {filename}")

            except Exception as e:
                raise HTTPException(400, f"Failed to load dataset: {str(e)}")

            # Prevent huge datasets
            if len(df) > 10000:
                df = df.head(10000)

            # Pickle dataset
            temp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
            temp_pkl.close()
            df.to_pickle(temp_pkl.name)
            pickle_path = temp_pkl.name

            df_preview = (
                f"\nDataset info:\n"
                f"- Rows: {len(df)}, Columns: {len(df.columns)}\n"
                f"- Columns: {', '.join(df.columns.astype(str))}\n"
                f"- Sample:\n{df.head(5).to_markdown(index=False)}\n"
            )

        # -------------------------
        # Build LLM rules
        # -------------------------
        if dataset_uploaded:
            llm_rules = (
                "Rules:\n"
                "1) Use provided DataFrame `df` only.\n"
                "2) DO NOT call scrape_url_to_dataframe().\n"
                "3) Return JSON with keys: questions, code.\n"
                "4) Code must fill `results` dict.\n"
                "5) Use plot_to_base64() for plots.\n"
            )
        else:
            llm_rules = (
                "Rules:\n"
                "1) If needed, call scrape_url_to_dataframe(url).\n"
                "2) Return JSON with keys: questions, code.\n"
                "3) Code must fill `results` dict.\n"
            )

        llm_input = (
            f"{llm_rules}\nQuestions:\n{raw_questions}\n"
            f"{df_preview}"
            "Respond with JSON only."
        )

        logger.info("LLM INPUT:\n%s", llm_input[:1000])

        # -------------------------
        # Run agent safely
        # -------------------------
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(run_agent_safely_unified, llm_input, pickle_path)

            try:
                result = fut.result(timeout=LLM_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                raise HTTPException(408, "Processing timeout")

        if not isinstance(result, dict):
            raise HTTPException(500, "Invalid agent result")

        if "error" in result:
            raise HTTPException(500, detail=result["error"])

        # -------------------------
        # Map keys + cast types
        # -------------------------
        if keys_list and type_map:
            mapped = {}

            for idx, (q, val) in enumerate(result.items()):
                if idx < len(keys_list):
                    key = keys_list[idx]
                    caster = type_map.get(key, str)

                    try:
                        if isinstance(val, str) and val.startswith("data:image/"):
                            val = val.split(",", 1)[1] if "," in val else val

                        mapped[key] = caster(val) if val not in (None, "") else val
                    except Exception:
                        mapped[key] = val

            result = mapped

        return JSONResponse(content=result)

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("analyze_data failed")
        raise HTTPException(500, detail=str(e))


def run_agent_safely_unified(llm_input: str, pickle_path: str = None) -> Dict:
    """
    Unified runner:
    - retries agent
    - parses JSON safely
    - handles scraping if needed
    - executes code
    """

    try:
        max_retries = 3
        raw_out = ""

        # -------------------------
        # Retry agent execution
        # -------------------------
        for attempt in range(1, max_retries + 1):
            response = agent_executor.invoke(
                {"input": llm_input},
                {"timeout": LLM_TIMEOUT_SECONDS}
            )

            raw_out = (
                response.get("output")
                or response.get("final_output")
                or response.get("text")
                or str(response)
            )

            if raw_out:
                break

            logger.warning(f"Attempt {attempt} returned empty output")

        if not raw_out:
            return {"error": f"No output after {max_retries} attempts"}

        logger.info("LLM RAW OUTPUT:\n%s", raw_out[:1000])

        # -------------------------
        # Parse JSON
        # -------------------------
        parsed = clean_llm_output(raw_out)

        if "error" in parsed:
            return parsed

        if not isinstance(parsed, dict):
            return {"error": "Parsed output is not dict", "raw": parsed}

        if "code" not in parsed or "questions" not in parsed:
            return {"error": "Invalid agent response format", "raw": parsed}

        code = parsed["code"]
        questions = parsed["questions"]

        logger.info("EXEC CODE:\n%s", code[:1000])

        # -------------------------
        # Basic safety filter
        # -------------------------
        if any(x in code for x in ["import os", "subprocess", "sys.exit", "open("]):
            return {"error": "Unsafe code detected"}

        # -------------------------
        # Handle scraping (if no dataset)
        # -------------------------
        if pickle_path is None:
            urls = list(set(re.findall(
                r"scrape_url_to_dataframe\(\s*['\"](.*?)['\"]\s*\)", code
            )))

            if urls:
                url = urls[0]

                tool_resp = scrape_url_to_dataframe(url)

                if tool_resp.get("status") != "success":
                    return {"error": f"Scrape failed: {tool_resp.get('message')}"}

                df = pd.DataFrame(tool_resp.get("data", []))

                if df.empty:
                    return {"error": "Scraped dataframe is empty"}

                temp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
                temp_pkl.close()

                df.to_pickle(temp_pkl.name)
                pickle_path = temp_pkl.name

        # -------------------------
        # Execute generated code
        # -------------------------
        exec_result = write_and_run_temp_python(
            code,
            injected_pickle=pickle_path,
            timeout=LLM_TIMEOUT_SECONDS
        )

        if exec_result.get("status") != "success":
            logger.error("Execution failed: %s", exec_result)
            return {
                "error": "Execution failed",
                "message": exec_result.get("message"),
                "raw": exec_result.get("raw")
            }

        results_dict = exec_result.get("result", {})

        if not isinstance(results_dict, dict):
            return {"error": "Execution returned invalid results", "raw": results_dict}

        # -------------------------
        # FIX: normalize mapping
        # -------------------------
        output = {}

        result_items = list(results_dict.items())

        for idx, q in enumerate(questions):
            try:
                # map by position instead of exact match
                _, val = result_items[idx]
                output[q] = val
            except Exception:
                output[q] = "Answer not found"

        return output

    except Exception as e:
        logger.exception("run_agent_safely_unified failed")
        return {"error": str(e)}




## -------------------------
# Favicon fallback
# -------------------------
_FAVICON_FALLBACK_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO3n+9QAAAAASUVORK5CYII="
)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """
    Serve favicon.ico if present, else return a tiny PNG.
    """
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, "favicon.ico")

        if os.path.exists(path):
            return FileResponse(path, media_type="image/x-icon")

        return Response(content=_FAVICON_FALLBACK_PNG, media_type="image/png")

    except Exception as e:
        logger.warning(f"Favicon error: {e}")
        return Response(content=_FAVICON_FALLBACK_PNG, media_type="image/png")


# -------------------------
# Health check endpoint
# -------------------------
@app.get("/api", include_in_schema=False)
async def analyze_get_info():
    """
    Health/info endpoint.
    """
    return JSONResponse({
        "ok": True,
        "service": "TDS Data Analyst Agent",
        "status": "running",
        "llm_mode": "grok" if USE_GROK else "gemini",
        "gemini_keys_loaded": len(GEMINI_KEYS),
        "message": "Use POST /api with 'questions_file' and optional 'data_file'."
    })



# -----------------------------
# System Diagnostics
# ----------------------------
# ---- Configuration for diagnostics (tweak as needed) ----
# ---- Configuration for diagnostics ----
DIAG_NETWORK_TARGETS = {
    "Google AI": "https://generativelanguage.googleapis.com",
    "AISTUDIO": "https://aistudio.google.com/",
    "OpenAI": "https://api.openai.com",
    "GitHub": "https://api.github.com",
}

DIAG_LLM_KEY_TIMEOUT = 20
DIAG_PARALLELISM = 4
RUN_LONGER_CHECKS = False


# -------------------------
# Safe access to globals
# -------------------------
_GEMINI_KEYS = globals().get("GEMINI_KEYS", [])
_MODEL_HIERARCHY = globals().get("MODEL_HIERARCHY", [])


# -------------------------
# Helpers
# -------------------------
def _now_iso():
    return datetime.utcnow().isoformat() + "Z"


_executor = ThreadPoolExecutor(max_workers=DIAG_PARALLELISM)

async def run_in_thread(fn, *a, timeout=30, **kw):
    loop = asyncio.get_running_loop()
    try:
        task = loop.run_in_executor(_executor, partial(fn, *a, **kw))
        return await asyncio.wait_for(task, timeout=timeout)
    except asyncio.TimeoutError:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)}


# -------------------------
# Checks
# -------------------------
def _env_check(required=None):
    required = required or []
    out = {}

    for k in required:
        val = os.getenv(k)
        out[k] = {
            "present": bool(val),
            "masked": (val[:4] + "..." + val[-4:]) if val else None
        }

    out["USE_GROK"] = os.getenv("USE_GROK")
    out["LLM_TIMEOUT_SECONDS"] = os.getenv("LLM_TIMEOUT_SECONDS")

    return out


def _system_info():
    try:
        return {
            "host": socket.gethostname(),
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "cpu_cores": psutil.cpu_count(logical=True),
            "memory_gb": round(psutil.virtual_memory().total / 1024**3, 2),
            "cwd_free_gb": round(shutil.disk_usage(os.getcwd()).free / 1024**3, 2),
            "tmp_free_gb": round(shutil.disk_usage(tempfile.gettempdir()).free / 1024**3, 2),
        }
    except Exception as e:
        return {"error": str(e)}


def _temp_write_test():
    try:
        path = os.path.join(tempfile.gettempdir(), f"diag_{int(time.time())}.tmp")
        with open(path, "w") as f:
            f.write("ok")
        os.remove(path)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _pandas_pipeline_test():
    try:
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        df["z"] = df["x"] * df["y"]
        return {"ok": True, "z_sum": int(df["z"].sum())}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _network_probe_sync(url):
    try:
        r = requests.head(url, timeout=10)
        return {"ok": True, "status": r.status_code}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _test_gemini_key_model(key, model):
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0,
            google_api_key=key
        )

        resp = llm.invoke("ping")
        text = str(resp)[:100] if resp else None

        return {"ok": True, "model": model, "response": text}

    except Exception as e:
        return {"ok": False, "error": str(e)}


# -------------------------
# Async wrappers
# -------------------------
async def check_network():
    tasks = [
        run_in_thread(_network_probe_sync, url)
        for url in DIAG_NETWORK_TARGETS.values()
    ]

    results = await asyncio.gather(*tasks)
    return dict(zip(DIAG_NETWORK_TARGETS.keys(), results))


async def check_llm_keys_models():
    if not _GEMINI_KEYS:
        return {"warning": "No Gemini keys configured"}

    for model in (_MODEL_HIERARCHY or ["gemini-2.5-flash"]):
        tasks = [
            run_in_thread(_test_gemini_key_model, key, model, timeout=DIAG_LLM_KEY_TIMEOUT)
            for key in _GEMINI_KEYS
        ]

        results = await asyncio.gather(*tasks)

        if any(r.get("ok") for r in results if isinstance(r, dict)):
            return {"model": model, "working": True}

    return {"working": False}


# -------------------------
# Diagnose endpoint
# -------------------------
@app.get("/summary")
async def diagnose(full: bool = Query(False)):
    started = datetime.utcnow()

    checks = {
        "env": run_in_thread(_env_check, ["GOOGLE_API_KEY"]),
        "system": run_in_thread(_system_info),
        "temp": run_in_thread(_temp_write_test),
        "pandas": run_in_thread(_pandas_pipeline_test),
        "network": asyncio.create_task(check_network()),
        "llm": asyncio.create_task(check_llm_keys_models())
    }

    if full:
        checks["duckdb"] = asyncio.create_task(check_duckdb())
        checks["playwright"] = asyncio.create_task(check_playwright())

    results = {}

    for name, task in checks.items():
        try:
            results[name] = await task
        except Exception as e:
            results[name] = {"error": str(e)}

    return {
        "status": "ok",
        "time": _now_iso(),
        "elapsed_sec": (datetime.utcnow() - started).total_seconds(),
        "checks": results
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
