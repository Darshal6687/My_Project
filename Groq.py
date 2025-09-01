import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import json
import os
import pandas as pd
import torch
from transformers import AutoTokenizer
from dotenv import load_dotenv
import requests
import re
import time 

st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #007bff;   /* Blue color */
        color: white;                /* White text */
        font-size: 20px;             /* Larger font */
        font-weight: bold;
        height: 30px;                /* Taller button */
        width: 190px;                /* Wider button */
        border-radius: 12px;
        border: none;
        display: block;
        margin: 20px auto;           /* Centers button horizontally */
    }
    div.stButton > button:first-child:hover {
        background-color: #0056b3;   /* Darker blue on hover */
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq()

# Load JSON files
with open("C:\\Job Planner\\Json\\drivers.json", "r") as f:
    drivers_json = json.load(f)

with open("C:\\Job Planner\\Json\\jobs.json", "r") as f:
    jobs_json = json.load(f)

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


# Streamlit UI
st.title("Job Planner Assignment")
st.write("Enter your prompt below to assign jobs to drivers:")

# User input
default_prompt = """
You are a job planner system. 
Always respond ONLY in valid JSON format with the following rules:

1. Output must be JSON only with fields: driver_id, jobs[].
2. Each job must include pickup_zone and dropoff_zone.
3. A driver's next job must have pickup_zone equal to the previous job's dropoff_zone.
4. Each driver can have at most 3 jobs.
5. Assign all jobs so none are left unassigned.
"""

user_prompt = st.text_area("Enter your prompt:", height=200, value=default_prompt)

# Force JSON response even if user doesn't mention it

# final_prompt = f"""
# You are a job planner system. 
# Your response MUST be in valid JSON format only.

# User request:
# {user_prompt}

# Jobs: {json.dumps(jobs_json)}
# Drivers: {json.dumps(drivers_json)}

# Return the assignment strictly in JSON format.
# """

final_prompt = f"""
You are a job planner system. 
Always return ONLY a valid JSON object in this format:

{{
  "assignments": [
    {{
      "driver_id": "DR-001",
      "jobs": [
        {{"job_id":"JOB-1001","pickup_zone":"z1","dropoff_zone":"z6"}}
      ]
    }}
  ]
}}

Jobs: {json.dumps(jobs_json)}
Drivers: {json.dumps(drivers_json)}

User Request: {user_prompt}
"""


if st.button("Run Job Assignment"):
    start_time = time.time()

    # --- Token Counting ---
    prompt_tokens = count_tokens_exact(final_prompt, model="gpt-oss:20b")
    st.info(f"📊 Token length: {prompt_tokens} tokens")

      # --- Display time placeholder ---
    time_placeholder = st.empty()  # reserve a space to update later
    with st.spinner("Generating job assignments..."):
        # Call Groq API
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
           messages=[
                {
                    "role": "user",
                    "content": final_prompt
                }
            ],
            temperature=1,
            max_completion_tokens=8192,
            top_p=1,
            reasoning_effort="medium",
            stream=False,
            response_format={"type": "json_object"},
        )

        # Extract response
        response_text = completion.choices[0].message.content

        try:
            response_json = json.loads(response_text)
            st.success("✅ Job assignment completed!")
           # st.json(response_json)  # Display nicely formatted JSON

           # Build table rows
            rows = []
            for driver in response_json.get("assignments", []):
                driver_id = driver.get("driver_id")
                for job in driver.get("jobs", []):
                    rows.append({
                        "Driver ID": driver_id,
                        "Job ID": job.get("job_id"),
                        "Pickup Zone": job.get("pickup_zone"),
                        "Dropoff Zone": job.get("dropoff_zone"),
                    })

            # Convert to DataFrame
            df = pd.DataFrame(rows)

            # Show rows with Assign button
            st.write("### Job Assignments")
            header_cols = st.columns([2, 2, 2, 2, 1])
            header_cols[0].write("**Driver ID**")
            header_cols[1].write("**Job ID**")
            header_cols[2].write("**Pickup Zone**")
            header_cols[3].write("**Dropoff Zone**")
            header_cols[4].write("**Action**")
            
            for i, row in df.iterrows():
                cols = st.columns([2, 2, 2, 2, 1])
                cols[0].write(row["Driver ID"])
                cols[1].write(row["Job ID"])
                cols[2].write(row["Pickup Zone"])
                cols[3].write(row["Dropoff Zone"])

                if cols[4].button("Assign", key=f"assign_{i}"):
                    st.success(f"✅ Assigned Job {row['Job ID']} to Driver {row['Driver ID']}")
        except Exception as e:
            st.error("⚠️ Failed to parse JSON output.")
            st.text(response_text)  # Show raw output


    end_time = time.time()     # record end
    duration = end_time - start_time
    if duration < 60:
        time_placeholder.info(f"⏱️ Task completed in {duration:.2f} seconds")
    else:
        minutes = duration / 60
        time_placeholder.info(f"⏱️ Task completed in {minutes:.2f} minutes")
# Assign jobs to drivers with the following algorithm:
# 1. Output must be JSON only with fields: driver_id, jobs[].
# 2. Include Pickup Zone and Dropoff Zone for every job.
# 3. A driver's next job must have Pickup Zone equal to the previous job's Dropoff Zone.
# 4. Each driver can have at most 3 jobs.
# 5. Assign all jobs so none are left unassigned.