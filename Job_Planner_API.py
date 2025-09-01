import streamlit as st
import json
import os
import requests
import re
import time 
import tiktoken
import torch
from transformers import AutoTokenizer
from dotenv import load_dotenv


load_dotenv()


# -------------------------
# Load JSON files on Windows
# @todo use Get api to get actual data of driver and job.
# convert to fake data.
# -------------------------


with open("C:\\Job Planner\\Json\\drivers.json", "r") as f:
    drivers_json = json.load(f)

with open("C:\\Job Planner\\Json\\jobs.json", "r") as f:
    jobs_json = json.load(f)

# -------------------------
# Helper function: Extract JSON
# -------------------------
# Instead of only extracting first {} or []
def extract_json(text):
    try:
        cleaned = text.strip()
        # Try if whole thing is JSON
        return json.loads(cleaned)
    except Exception:
        # Fallback to regex search
        match = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
    return None

# -------------------------
# Helper: Count Tokens
# -------------------------
# Cache tokenizers per model
tokenizer_cache = {}

def count_tokens_exact(text, model="gpt-oss:20b"):
    global tokenizer_cache
    
    if model not in tokenizer_cache:
        # Map Ollama model to HuggingFace equivalent
        if "llama" in model:
            hf_model = "meta-llama/Llama-2-7b-hf"   # adjust for llama3.2 once available
        elif "gpt-oss" in model:
            hf_model = "EleutherAI/gpt-neox-20b"    # matches gpt-oss:20b arch
        else:
            hf_model = "gpt2"  # fallback
        
        tokenizer_cache[model] = AutoTokenizer.from_pretrained(hf_model)
    
    tokenizer = tokenizer_cache[model]
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


# -------------------------
# Streamlit UI
# -------------------------
st.title("📂 Job Planner")
st.write("Enter your prompt below to assign jobs to drivers:")

default_prompt = """
You are a job planner system. 
Always respond ONLY in valid JSON format with the following rules:

1. Output must be JSON only with fields: driver_id, jobs[].
2. Each job must include pickup_zone and dropoff_zone.
3. A driver's next job must have pickup_zone equal to the previous job's dropoff_zone.
4. Each driver can have at most 3 jobs.
5. Assign all jobs so none are left unassigned.
6. Do not include explanations or text outside the JSON.
"""

user_prompt = st.text_area("Enter your prompt:", height=200, value=default_prompt)

final_prompt = f"""
Act as a job planner system.
Your ONLY output must be a valid JSON array.

User request:
{user_prompt}

Follow this schema exactly:
[
  {{
    "driver_id": "DR-001",
    "jobs": [
      {{
        "job_id": "JOB-1001",
        "pickup_zone": "z1",
        "dropoff_zone": "z6"
      }},
      {{
        "job_id": "JOB-1005",
        "pickup_zone": "z6",
        "dropoff_zone": "z9"
      }}
    ]
  }}
]
Jobs: {json.dumps(jobs_json)}
Drivers: {json.dumps(drivers_json)}
Return the assignment strictly in JSON format.  
"""


if st.button("Run Job Assignment"):
    start_time = time.time()

    # --- Token Counting ---
    prompt_tokens = count_tokens_exact(final_prompt, model="gpt-oss:20b")
    st.info(f"📊 Token length: {prompt_tokens} tokens")

      # --- Display time placeholder ---
    time_placeholder = st.empty()  # reserve a space to update later

    with st.spinner("Connecting to Ollama..."):
        try:
            OLLAMA_API = os.getenv("OLLAMA_API")

            payload = {
                "model": "gpt-oss:20b",  #  "llama3.2:latest" "gpt-oss:20b"
                "messages": [
                   {"role": "system", "content": "Return only a JSON array following the provided schema. No explanations, no alternative keys."},
                    {"role": "user", "content": final_prompt}
                ],
                "stream": True  # <-- IMPORTANT for Ollama
            }
            
            # Collect streamed response
            response = requests.post(OLLAMA_API, json=payload, stream=True)
            response.raise_for_status()

            collected_text = ""
            for line in response.iter_lines():
                if line:
                    try:
                        obj = json.loads(line.decode("utf-8"))
                        # Append message content if available
                        if "message" in obj and "content" in obj["message"]:
                            collected_text += obj["message"]["content"]
                    except Exception:
                        pass

            # Try extracting JSON
            response_json = extract_json(collected_text)

            if response_json:
                st.success("✅ Job assignment completed!")
                st.json(response_json)
            else:
                st.error("⚠️ No JSON object found in response.")
                st.text(collected_text)
    
        except Exception as e:
            st.error(f"❌ API error: {str(e)}")

    end_time = time.time()     # record end
    duration = end_time - start_time
    if duration < 60:
        time_placeholder.info(f"⏱️ Task completed in {duration:.2f} seconds")
    else:
        minutes = duration / 60
        time_placeholder.info(f"⏱️ Task completed in {minutes:.2f} minutes")



    


