import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import json
import pandas as pd
import time
from transformers import AutoTokenizer
import re

# ----------------- Custom CSS -----------------
st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #007bff;
        color: white;
        font-size: 20px;
        font-weight: bold;
        height: 40px;
        width: 190px;
        border-radius: 12px;
        border: none;
        display: block;
        margin: 20px auto;
    }
    div.stButton > button:first-child:hover {
        background-color: #0056b3;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------- Load env & client -----------------
load_dotenv()
client = Groq()

# ----------------- File Uploaders -----------------
st.title("📂 Job Planner")

st.markdown("### Upload Driver and Job CSVs")

# Load sample files
drivers_sample = pd.read_csv("resources/Drivers.csv")
jobs_sample = pd.read_csv("resources/Jobs.csv")

# Add download buttons
st.markdown("#### 📥 Download sample files")
st.download_button(
    label="Download Drivers.csv",
    data=drivers_sample.to_csv(index=False),
    file_name="Drivers.csv",
    mime="text/csv",
)
st.download_button(
    label="Download Jobs.csv",
    data=jobs_sample.to_csv(index=False),
    file_name="Jobs.csv",
    mime="text/csv",
)

drivers_file = st.file_uploader("Upload drivers.csv", type=["csv"])
jobs_file = st.file_uploader("Upload jobs.csv", type=["csv"])

if drivers_file is not None and jobs_file is not None:
    drivers_df = pd.read_csv(drivers_file)
    jobs_df = pd.read_csv(jobs_file)

    if "driver_id" in drivers_df.columns:
        drivers_json = drivers_df[["driver_id"]].to_dict(orient="records")
    else:
        st.error("⚠️ drivers.csv must contain a 'driver_id' column")
        st.stop()

    # Normalize column names
    jobs_df.columns = [c.strip().lower() for c in jobs_df.columns]
    col_map = {
        "job_id": "job_id",
        "pickup zone": "pickup_zone",
        "dropoff zone": "dropoff_zone"
    }
    missing = [c for c in col_map.keys() if c not in jobs_df.columns]
    if missing:
        st.error(f"⚠️ jobs.csv must contain columns: {', '.join(col_map.keys())}")
        st.stop()

    jobs_df = jobs_df.rename(columns=col_map)
    jobs_json = jobs_df[["job_id", "pickup_zone", "dropoff_zone"]].to_dict(orient="records")

    # ----------------- Token Counter -----------------
    tokenizer_cache = {}
    def count_tokens_exact(text, model="gpt-oss:20b"):
        global tokenizer_cache
        if model not in tokenizer_cache:
            hf_model = "EleutherAI/gpt-neox-20b"
            tokenizer_cache[model] = AutoTokenizer.from_pretrained(hf_model)
        tokenizer = tokenizer_cache[model]
        return len(tokenizer.encode(text, add_special_tokens=False))

    # ----------------- Prompt -----------------
    default_prompt = "Assign jobs to drivers"

    user_prompt = st.text_area("Enter your prompt:", height=200, value=default_prompt)

    final_prompt = f"""
    Return ONLY a valid JSON object.
    Do not include explanations.
    Do not include markdown.
    Do not include reasoning.
    Do not include text before or after.
    The response MUST be a single JSON object that matches the required format.

    Format:
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

    Rules:
    - Each driver max 3 jobs.
    - Next pickup_zone must equal previous dropoff_zone.
    - Assign ALL jobs so none are left unassigned.

    Data:
    Drivers = {drivers_json}
    Jobs = {jobs_json}

    User Request: {user_prompt}
    """

    # ----------------- Run Assignment -----------------
    if st.button("Run Job Assignment"):
        start_time = time.time()

        prompt_tokens = count_tokens_exact(final_prompt, model="openai/gpt-oss-20b")
        st.info(f"📊 Token length: {prompt_tokens} tokens")

        # --- Display time placeholder ---
        time_placeholder = st.empty()  # reserve a space to update later


        with st.spinner("Generating job assignments..."):
            completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",   # ✅ recommended replacement  openai/gpt-oss-20b  qwen/qwen3-32b
            messages=[
                {"role": "system", "content": "You are a JSON generator. Return ONLY valid JSON. No reasoning. No text. No markdown."},
                {"role": "user", "content": final_prompt},
            ],
            temperature=0,             # ✅ deterministic
            max_tokens=8192,           # ✅ keep smaller, avoids truncation
            top_p=1,
            stream=False,
            response_format={"type": "json_object"},
        )

            # ---------- JSON Cleaning ----------
            response_text = completion.choices[0].message.content
            if not response_text or not response_text.strip():
                st.error("⚠️ Model returned empty response")
                st.json(completion.model_dump())  # 🔍 Debug entire API response
                st.stop()

            #st.text_area("🔍 Raw model output:", value=response_text, height=200)

            # Strip all Markdown fences, if present
            clean_text = re.sub(r"^```[a-zA-Z]*", "", response_text.strip(), flags=re.IGNORECASE|re.MULTILINE)
            clean_text = re.sub(r"```$", "", clean_text.strip(), flags=re.MULTILINE).strip()

            response_json = None
            try:
                response_json = json.loads(clean_text)
            except json.JSONDecodeError:
                # fallback: try to find any JSON object inside the text
                match = re.search(r"\{[\s\S]*\}", response_text)
                if match:
                    try:
                        response_json = json.loads(match.group(0))
                    except json.JSONDecodeError:
                        pass

            if not response_json:
                st.error("⚠️ No JSON object found in model output")
                st.text_area("🚨 Debug raw response:", value=response_text, height=200)
                st.stop()

            # ---------- Display Results ----------
            st.success("✅ Job assignment completed!")
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

            df = pd.DataFrame(rows)
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

        duration = time.time() - start_time
        if duration < 60:
            time_placeholder.info(f"⏱️ Completed in {duration:.2f} seconds")
        else:
            minutes = duration/60
            time_placeholder.info(f"Completed in {minutes:.2f} minutes") 


