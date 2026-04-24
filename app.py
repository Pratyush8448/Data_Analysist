import os
import re
import json
import base64
import tempfile
import sys
import subprocess
import logging
import asyncio
import httpx
import importlib.metadata
import traceback
import socket
import platform
import shutil
import time

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from io import BytesIO, StringIO
from typing import Dict, Any, List

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import psutil

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, Response
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TDS Data Analyst Agent")

# -------------------- Gemini config --------------------
GEMINI_KEYS = [os.getenv(f"gemini_api_{i}") for i in range(1, 11)]
GEMINI_KEYS = [k for k in GEMINI_KEYS if k]

MODEL_HIERARCHY = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
]

QUOTA_KEYWORDS = ["quota", "exceeded", "rate limit", "403", "too many requests"]
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", 240))

if not GEMINI_KEYS:
    raise RuntimeError("No Gemini API keys found. Please set them in your environment.")


# -------------------- LLM wrapper --------------------
class LLMWithFallback:
    def __init__(self, keys=None, models=None, temperature=0):
        self.keys = keys or GEMINI_KEYS
        self.models = models or MODEL_HIERARCHY
        self.temperature = temperature
        self.slow_keys_log = defaultdict(list)
        self.failing_keys_log = defaultdict(int)

    def _get_llm_instance(self):
        last_error = None
        for model in self.models:
            for key in self.keys:
                try:
                    return ChatGoogleGenerativeAI(
                        model=model,
                        temperature=self.temperature,
                        google_api_key=key,
                    )
                except Exception as e:
                    last_error = e
                    msg = str(e).lower()
                    if any(qk in msg for qk in QUOTA_KEYWORDS):
                        self.slow_keys_log[key].append(model)
                    self.failing_keys_log[key] += 1
                    time.sleep(0.5)
        raise RuntimeError(f"All models/keys failed. Last error: {last_error}")

    def bind_tools(self, tools):
        return self._get_llm_instance().bind_tools(tools)

    def invoke(self, prompt):
        return self._get_llm_instance().invoke(prompt)


llm = LLMWithFallback(temperature=0)


# -------------------- Helpers --------------------
def parse_keys_and_types(questions_text: str):
    """Parse key/type annotations from questions file."""
    pattern = r"-\s*`([^`]+)`\s*:\s*(\w+)"
    matches = re.findall(pattern, questions_text)
    type_map_def = {
        "number": float,
        "string": str,
        "integer": int,
        "int": int,
        "float": float,
    }
    type_map = {key: type_map_def.get(t.lower(), str) for key, t in matches}
    keys_list = [k for k, _ in matches]
    return keys_list, type_map


def clean_llm_output(output: str) -> Dict:
    """Extract JSON object from LLM output robustly."""
    try:
        if not output:
            return {"error": "Empty LLM output"}
        s = re.sub(r"^```(?:json)?\s*", "", output.strip())
        s = re.sub(r"\s*```$", "", s)
        first = s.find("{")
        last = s.rfind("}")
        if first == -1 or last == -1 or last <= first:
            return {"error": "No JSON object found in LLM output", "raw": s}
        candidate = s[first : last + 1]
        try:
            return json.loads(candidate)
        except Exception as e:
            for i in range(last, first, -1):
                cand = s[first : i + 1]
                try:
                    return json.loads(cand)
                except Exception:
                    continue
            return {"error": f"JSON parsing failed: {str(e)}", "raw": candidate}
    except Exception as e:
        return {"error": str(e)}


# -------------------- Scraper (plain function + LangChain tool) --------------------
def _scrape_url_impl(url: str) -> Dict[str, Any]:
    """
    Core scraping logic used by both the LangChain tool and direct calls.
    Returns {"status": "success", "data": [...], "columns": [...]} or {"status": "error", ...}
    """
    try:
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/138.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.google.com/",
        }

        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "").lower()
        df = None

        if "text/csv" in ctype or url.lower().endswith(".csv"):
            df = pd.read_csv(BytesIO(resp.content))
        elif any(url.lower().endswith(ext) for ext in (".xls", ".xlsx")) or "spreadsheetml" in ctype:
            df = pd.read_excel(BytesIO(resp.content))
        elif url.lower().endswith(".parquet"):
            df = pd.read_parquet(BytesIO(resp.content))
        elif "application/json" in ctype or url.lower().endswith(".json"):
            try:
                data = resp.json()
                df = pd.json_normalize(data)
            except Exception:
                df = pd.DataFrame([{"text": resp.text}])
        elif "text/html" in ctype or re.search(r"/wiki/|\.org|\.com", url, re.IGNORECASE):
            try:
                tables = pd.read_html(StringIO(resp.text), flavor="bs4")
                if tables:
                    df = tables[0]
            except ValueError:
                pass
            if df is None:
                soup = BeautifulSoup(resp.text, "html.parser")
                text = soup.get_text(separator="\n", strip=True)
                df = pd.DataFrame({"text": [text]})
        else:
            df = pd.DataFrame({"text": [resp.text]})

        df.columns = df.columns.map(str).str.replace(r"\[.*\]", "", regex=True).str.strip()

        return {
            "status": "success",
            "data": df.to_dict(orient="records"),
            "columns": df.columns.tolist(),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@tool
def scrape_url_to_dataframe(url: str) -> Dict[str, Any]:
    """
    Fetch a URL and return data as a DataFrame.
    Supports HTML tables, CSV, Excel, Parquet, JSON, and plain text.
    Returns {"status": "success", "data": [...], "columns": [...]} on success.
    """
    return _scrape_url_impl(url)


# -------------------- SCRAPE_FUNC injected into sandboxed scripts --------------------
SCRAPE_FUNC = r'''
from typing import Dict, Any
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

def scrape_url_to_dataframe(url: str) -> Dict[str, Any]:
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        response.raise_for_status()
    except Exception as e:
        return {"status": "error", "error": str(e), "data": [], "columns": []}

    try:
        tables = pd.read_html(response.text)
        if tables:
            df = tables[0]
            df.columns = [str(c).strip() for c in df.columns]
            return {
                "status": "success",
                "data": df.to_dict(orient="records"),
                "columns": list(df.columns),
            }
    except Exception:
        pass

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")
    text_data = soup.get_text(separator="\n", strip=True)
    return {
        "status": "success",
        "data": [{"text": text_data}],
        "columns": ["text"],
    }
'''


# -------------------- Sandboxed code execution --------------------
def write_and_run_temp_python(
    code: str, injected_pickle: str = None, timeout: int = 60
) -> Dict[str, Any]:
    """
    Execute user-supplied code in a subprocess sandbox.
    Optionally injects a pickled DataFrame as `df` and `data`.
    Returns {"status": "success", "result": {...}} or {"status": "error", "message": "..."}.
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
        preamble.append(f"df = pd.read_pickle(r'''{injected_pickle}''')\n")
        preamble.append("data = df.to_dict(orient='records')\n")
    else:
        preamble.append("df = None\n")
        preamble.append("data = {}\n")

    # FIX: plot_to_base64 now returns a data-URI string with correct MIME type
    # so callers always know what format they received.
    helper = r'''
def plot_to_base64(max_bytes=100000):
    """Return a base64 data-URI string (image/png or image/webp) under max_bytes."""
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_bytes = buf.getvalue()
    if len(img_bytes) <= max_bytes:
        return "data:image/png;base64," + base64.b64encode(img_bytes).decode('ascii')
    for dpi in [80, 60, 50, 40, 30]:
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
        buf.seek(0)
        b = buf.getvalue()
        if len(b) <= max_bytes:
            return "data:image/png;base64," + base64.b64encode(b).decode('ascii')
    try:
        from PIL import Image
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=40)
        buf.seek(0)
        im = Image.open(buf)
        for quality in [80, 60]:
            out_buf = BytesIO()
            im.save(out_buf, format='WEBP', quality=quality, method=6)
            out_buf.seek(0)
            ob = out_buf.getvalue()
            if len(ob) <= max_bytes:
                return "data:image/webp;base64," + base64.b64encode(ob).decode('ascii')
    except Exception:
        pass
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=20)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('ascii')
'''

    script_lines = []
    script_lines.extend(preamble)
    script_lines.append(helper)
    script_lines.append(SCRAPE_FUNC)
    script_lines.append("\nresults = {}\n")
    script_lines.append(code)
    script_lines.append(
        "\nprint(json.dumps({'status':'success','result':results}, default=str), flush=True)\n"
    )

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    )
    tmp.write("\n".join(script_lines))
    tmp.flush()
    tmp_path = tmp.name
    tmp.close()

    try:
        completed = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if completed.returncode != 0:
            return {
                "status": "error",
                "message": completed.stderr.strip() or completed.stdout.strip(),
            }
        out = completed.stdout.strip()
        try:
            return json.loads(out)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Could not parse JSON output: {str(e)}",
                "raw": out,
            }
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Execution timed out"}
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        if injected_pickle:
            try:
                os.unlink(injected_pickle)
            except Exception:
                pass


# -------------------- LangChain agent setup --------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a full-stack autonomous data analyst agent.

You will receive:
- A set of **rules** for this request
- One or more **questions**
- An optional **dataset preview**

You must:
1. Follow the provided rules exactly.
2. Return only a valid JSON object — no extra commentary or formatting.
3. The JSON must contain:
   - "questions": [ list of original question strings exactly as provided ]
   - "code": "..." (Python code that creates a dict called `results` with each question string as a key and its computed answer as the value)
4. Your Python code will run in a sandbox with:
   - pandas, numpy, matplotlib available
   - A helper function `plot_to_base64(max_bytes=100000)` that returns a data-URI string (e.g. "data:image/png;base64,...").
5. When returning plots, always store the full data-URI returned by `plot_to_base64()` as the value.
6. Make sure all variables are defined before use.
""",
    ),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(
    llm=llm,
    tools=[scrape_url_to_dataframe],
    prompt=prompt,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[scrape_url_to_dataframe],
    verbose=True,
    max_iterations=5,       # increased so tool results can be fed back
    early_stopping_method="generate",
    handle_parsing_errors=True,
    return_intermediate_steps=True,  # FIX: capture tool call results
)


# -------------------- Core runner --------------------
def run_agent_safely_unified(llm_input: str, pickle_path: str = None) -> Dict:
    """
    1. Invoke the LangChain agent (up to 3 retries on empty output).
    2. Parse JSON from the agent's final output.
    3. If the agent didn't call the scrape tool itself and the generated code references it,
       run the scrape here and inject the resulting pickle.
    4. Execute the generated code in a subprocess sandbox.
    5. Return a dict mapping question strings to their answers.
    """
    try:
        raw_out = ""
        for attempt in range(1, 4):
            response = agent_executor.invoke(
                {"input": llm_input}, {"timeout": LLM_TIMEOUT_SECONDS}
            )
            raw_out = (
                response.get("output")
                or response.get("final_output")
                or response.get("text")
                or ""
            )
            if raw_out:
                break
            logger.warning("Agent returned empty output (attempt %d/3)", attempt)

        if not raw_out:
            return {"error": "Agent returned no output after 3 attempts"}

        parsed = clean_llm_output(raw_out)
        if "error" in parsed:
            return parsed

        if "code" not in parsed or "questions" not in parsed:
            return {"error": f"Invalid agent response format: {parsed}"}

        code: str = parsed["code"]
        questions: List[str] = parsed["questions"]

        # If no pickle was pre-loaded and the generated code calls the scraper,
        # run the scrape now and inject the result.
        if pickle_path is None:
            urls = re.findall(
                r"scrape_url_to_dataframe\(\s*['\"](.+?)['\"]\s*\)", code
            )
            if urls:
                # FIX: call the plain implementation, not the LangChain tool wrapper
                tool_resp = _scrape_url_impl(urls[0])
                if tool_resp.get("status") != "success":
                    return {
                        "error": f"Scrape tool failed: {tool_resp.get('message', tool_resp)}"
                    }
                df = pd.DataFrame(tool_resp["data"])
                tmp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
                tmp_pkl.close()
                df.to_pickle(tmp_pkl.name)
                pickle_path = tmp_pkl.name

        exec_result = write_and_run_temp_python(
            code, injected_pickle=pickle_path, timeout=LLM_TIMEOUT_SECONDS
        )
        if exec_result.get("status") != "success":
            return {
                "error": f"Execution failed: {exec_result.get('message')}",
                "raw": exec_result.get("raw"),
            }

        results_dict = exec_result.get("result", {})
        return {q: results_dict.get(q, "Answer not found") for q in questions}

    except Exception as e:
        logger.exception("run_agent_safely_unified failed")
        return {"error": str(e)}


# -------------------- Routes --------------------
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Frontend not found</h1><p>Place index.html alongside app.py</p>",
            status_code=404,
        )


@app.get("/api", include_in_schema=False)
async def analyze_get_info():
    return JSONResponse(
        {
            "ok": True,
            "message": "Server is running. Use POST /api with a questions file and optional data file.",
        }
    )


@app.post("/api")
async def analyze_data(request: Request):
    try:
        form = await request.form()
        questions_file = None
        data_file = None

        for _key, val in form.items():
            if not hasattr(val, "filename") or not val.filename:
                continue
            fname = val.filename.lower()
            # FIX: explicit field-name check; fall back to extension
            if fname.endswith(".txt") and questions_file is None:
                questions_file = val
            elif data_file is None:
                data_file = val

        if not questions_file:
            raise HTTPException(400, "Missing questions file (.txt)")

        raw_questions = (await questions_file.read()).decode("utf-8")
        keys_list, type_map = parse_keys_and_types(raw_questions)

        pickle_path = None
        df_preview = ""
        dataset_uploaded = False

        if data_file:
            dataset_uploaded = True
            filename = data_file.filename.lower()
            content = await data_file.read()

            if filename.endswith(".csv"):
                df = pd.read_csv(BytesIO(content))
            elif filename.endswith((".xlsx", ".xls")):
                df = pd.read_excel(BytesIO(content))
            elif filename.endswith(".parquet"):
                df = pd.read_parquet(BytesIO(content))
            elif filename.endswith(".json"):
                try:
                    df = pd.read_json(BytesIO(content))
                except ValueError:
                    df = pd.DataFrame(json.loads(content.decode("utf-8")))
            elif filename.endswith((".png", ".jpg", ".jpeg")):
                # FIX: guard PIL usage with PIL_AVAILABLE
                if not PIL_AVAILABLE:
                    raise HTTPException(400, "PIL not available for image processing")
                try:
                    image = Image.open(BytesIO(content)).convert("RGB")
                    df = pd.DataFrame({"image": [image]})
                except Exception as e:
                    raise HTTPException(400, f"Image processing failed: {str(e)}")
            else:
                raise HTTPException(400, f"Unsupported data file type: {filename}")

            tmp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
            tmp_pkl.close()
            df.to_pickle(tmp_pkl.name)
            pickle_path = tmp_pkl.name

            df_preview = (
                f"\n\nThe uploaded dataset has {len(df)} rows and {len(df.columns)} columns.\n"
                f"Columns: {', '.join(df.columns.astype(str))}\n"
                f"First rows:\n{df.head(5).to_markdown(index=False)}\n"
            )

        if dataset_uploaded:
            llm_rules = (
                "Rules:\n"
                "1) You have access to a pandas DataFrame called `df` and its dict form `data`.\n"
                "2) DO NOT call scrape_url_to_dataframe() or fetch any external data.\n"
                "3) Use only the uploaded dataset.\n"
                "4) Produce a JSON object with keys:\n"
                '   - "questions": [ ... original question strings ... ]\n'
                '   - "code": "..."  (Python that fills `results` with question strings as keys)\n'
                "5) For plots: use plot_to_base64() — it returns a data-URI string.\n"
            )
        else:
            llm_rules = (
                "Rules:\n"
                "1) If you need web data, call scrape_url_to_dataframe(url) in your code.\n"
                "2) Produce a JSON object with keys:\n"
                '   - "questions": [ ... original question strings ... ]\n'
                '   - "code": "..."  (Python that fills `results` with question strings as keys)\n'
                "3) For plots: use plot_to_base64() — it returns a data-URI string.\n"
            )

        llm_input = (
            f"{llm_rules}\nQuestions:\n{raw_questions}\n"
            f"{df_preview}"
            "Respond with the JSON object only."
        )

        # Run in a thread pool so the async event loop is not blocked
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            future = loop.run_in_executor(
                pool,
                partial(run_agent_safely_unified, llm_input, pickle_path),
            )
            try:
                result = await asyncio.wait_for(future, timeout=LLM_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                raise HTTPException(408, "Processing timeout")

        if "error" in result:
            raise HTTPException(500, detail=result["error"])

        # Post-process: map positional keys and cast types
        if keys_list and type_map:
            mapped = {}
            for idx, (q, val) in enumerate(result.items()):
                if idx < len(keys_list):
                    key = keys_list[idx]
                    caster = type_map.get(key, str)
                    try:
                        # FIX: strip data-URI prefix for image values before returning
                        if isinstance(val, str) and val.startswith("data:image/"):
                            mapped[key] = val  # preserve full data-URI
                        else:
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


# -------------------- Diagnostics --------------------
DIAG_NETWORK_TARGETS = {
    "Google AI": "https://generativelanguage.googleapis.com",
    "AISTUDIO": "https://aistudio.google.com/",
    "OpenAI": "https://api.openai.com",
    "GitHub": "https://api.github.com",
}
DIAG_LLM_KEY_TIMEOUT = 30
DIAG_PARALLELISM = 6
RUN_LONGER_CHECKS = False

_executor = ThreadPoolExecutor(max_workers=DIAG_PARALLELISM)


def _now_iso():
    return datetime.utcnow().isoformat() + "Z"


async def run_in_thread(fn, *a, timeout=30, **kw):
    loop = asyncio.get_running_loop()
    task = loop.run_in_executor(_executor, partial(fn, *a, **kw))
    return await asyncio.wait_for(task, timeout=timeout)


def _env_check(required=None):
    required = required or []
    out = {}
    for k in required:
        v = os.getenv(k)
        out[k] = {
            "present": bool(v),
            "masked": (v[:4] + "..." + v[-4:]) if v else None,
        }
    out["GOOGLE_MODEL"] = os.getenv("GOOGLE_MODEL")
    out["LLM_TIMEOUT_SECONDS"] = os.getenv("LLM_TIMEOUT_SECONDS")
    return out


def _system_info():
    info = {
        "host": socket.gethostname(),
        "platform": platform.system(),
        "platform_release": platform.release(),
        "python_version": platform.python_version(),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "memory_total_gb": round(psutil.virtual_memory().total / 1024**3, 2),
    }
    try:
        info["cwd_free_gb"] = round(shutil.disk_usage(os.getcwd()).free / 1024**3, 2)
    except Exception:
        info["cwd_free_gb"] = None
    try:
        info["tmp_free_gb"] = round(
            shutil.disk_usage(tempfile.gettempdir()).free / 1024**3, 2
        )
    except Exception:
        info["tmp_free_gb"] = None
    try:
        import torch
        info["torch_installed"] = True
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch_installed"] = False
        info["cuda_available"] = False
    return info


def _temp_write_test():
    path = os.path.join(tempfile.gettempdir(), f"diag_test_{int(time.time())}.tmp")
    with open(path, "w") as f:
        f.write("ok")
    ok = os.path.exists(path)
    os.remove(path)
    return {"tmp_dir": tempfile.gettempdir(), "write_ok": ok}


def _app_write_test():
    cwd = os.getcwd()
    path = os.path.join(cwd, f"diag_test_{int(time.time())}.tmp")
    with open(path, "w") as f:
        f.write("ok")
    ok = os.path.exists(path)
    os.remove(path)
    return {"cwd": cwd, "write_ok": ok}


def _pandas_pipeline_test():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    df["z"] = df["x"] * df["y"]
    return {"rows": df.shape[0], "cols": df.shape[1], "z_sum": int(df["z"].sum())}


def _installed_packages_sample():
    try:
        pkgs = []
        for dist in importlib.metadata.distributions():
            try:
                pkgs.append(f"{dist.metadata['Name']}=={dist.version}")
            except Exception:
                continue
        return {"sample_packages": sorted(pkgs)[:20]}
    except Exception as e:
        return {"error": str(e)}


def _network_probe_sync(url, timeout=30):
    try:
        r = requests.head(url, timeout=timeout)
        return {
            "ok": True,
            "status_code": r.status_code,
            "latency_ms": int(r.elapsed.total_seconds() * 1000),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _test_gemini_key_model(key, model, ping_text="ping"):
    try:
        obj = ChatGoogleGenerativeAI(
            model=model, temperature=0, google_api_key=key
        )

        def _text(resp):
            if resp is None:
                return None
            if isinstance(resp, str):
                return resp
            if hasattr(resp, "content") and isinstance(resp.content, str):
                return resp.content
            if hasattr(resp, "text") and isinstance(resp.text, str):
                return resp.text
            return str(resp)

        try:
            resp = obj.invoke(ping_text)
            return {"ok": True, "model": model, "summary": (_text(resp) or "")[:200]}
        except Exception as e_invoke:
            try:
                resp = obj(ping_text)
                return {"ok": True, "model": model, "summary": (_text(resp) or "")[:200]}
            except Exception as e_call:
                return {
                    "ok": False,
                    "error": f"invoke: {e_invoke}; call: {e_call}",
                }
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def check_network():
    tasks = {
        name: asyncio.create_task(run_in_thread(_network_probe_sync, url, timeout=30))
        for name, url in DIAG_NETWORK_TARGETS.items()
    }
    out = {}
    for name, task in tasks.items():
        try:
            out[name] = await task
        except Exception as e:
            out[name] = {"ok": False, "error": str(e)}
    return out


async def check_llm_keys_models():
    if not GEMINI_KEYS:
        return {"warning": "no GEMINI_KEYS configured"}
    results = []
    for model in MODEL_HIERARCHY:
        tasks = [
            asyncio.create_task(
                run_in_thread(_test_gemini_key_model, k, model, timeout=DIAG_LLM_KEY_TIMEOUT)
            )
            for k in GEMINI_KEYS
        ]
        completed = await asyncio.gather(*tasks, return_exceptions=True)
        summary = {"model": model, "attempts": []}
        any_ok = False
        for key, res in zip(GEMINI_KEYS, completed):
            mask = (key[:4] + "..." + key[-4:]) if key else None
            if isinstance(res, Exception):
                summary["attempts"].append({"key_mask": mask, "ok": False, "error": str(res)})
            else:
                summary["attempts"].append({"key_mask": mask, **res})
                if res.get("ok"):
                    any_ok = True
        results.append(summary)
        if any_ok:
            break
    return {"models_tested": results}


async def check_duckdb():
    try:
        import duckdb

        def duck_check():
            conn = duckdb.connect(":memory:")
            conn.execute("SELECT 1")
            conn.close()
            return {"duckdb": True}

        return await run_in_thread(duck_check, timeout=30)
    except Exception as e:
        return {"duckdb_error": str(e)}


async def check_playwright():
    try:
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            b = await p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = await b.new_page()
            await page.goto("about:blank")
            ua = await page.evaluate("() => navigator.userAgent")
            await b.close()
            return {"playwright_ok": True, "ua": ua[:200]}
    except Exception as e:
        return {"playwright_error": str(e)}


@app.get("/summary")
async def diagnose(
    full: bool = Query(False, description="Run extended checks (duckdb/playwright)")
):
    started = datetime.utcnow()
    report: Dict[str, Any] = {
        "status": "ok",
        "server_time": _now_iso(),
        "summary": {},
        "checks": {},
        "elapsed_seconds": None,
    }

    tasks = {
        "env": run_in_thread(
            _env_check,
            ["GOOGLE_API_KEY", "GOOGLE_MODEL", "LLM_TIMEOUT_SECONDS"],
            timeout=3,
        ),
        "system": run_in_thread(_system_info, timeout=30),
        "tmp_write": run_in_thread(_temp_write_test, timeout=30),
        "cwd_write": run_in_thread(_app_write_test, timeout=30),
        "pandas": run_in_thread(_pandas_pipeline_test, timeout=30),
        "packages": run_in_thread(_installed_packages_sample, timeout=50),
        "network": asyncio.create_task(check_network()),
        "llm_keys_models": asyncio.create_task(check_llm_keys_models()),
    }

    if full or RUN_LONGER_CHECKS:
        tasks["duckdb"] = asyncio.create_task(check_duckdb())
        tasks["playwright"] = asyncio.create_task(check_playwright())

    results = {}
    for name, coro in tasks.items():
        try:
            results[name] = {"status": "ok", "result": await coro}
        except TimeoutError:
            results[name] = {"status": "timeout", "error": "check timed out"}
        except Exception as e:
            results[name] = {
                "status": "error",
                "error": str(e),
                "trace": traceback.format_exc(),
            }

    report["checks"] = results
    failed = [k for k, v in results.items() if v.get("status") != "ok"]
    report["status"] = "warning" if failed else "ok"
    report["summary"]["failed_checks"] = failed
    report["elapsed_seconds"] = (datetime.utcnow() - started).total_seconds()
    return report


# -------------------- Favicon --------------------
_FAVICON_FALLBACK = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO3n+9QAAAAASUVORK5CYII="
)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    if os.path.exists("favicon.ico"):
        return FileResponse("favicon.ico", media_type="image/x-icon")
    return Response(content=_FAVICON_FALLBACK, media_type="image/png")


# -------------------- Entry point --------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
