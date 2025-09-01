import streamlit as st
import json
import ollama

# Load JSON files
with open("C:\\Job Planner\\Json\\drivers.json", "r") as f:
    drivers_json = json.load(f)

with open("C:\\Job Planner\\Json\\jobs.json", "r") as f:
    jobs_json = json.load(f)

# Streamlit UI
st.write("Enter your prompt below to assign jobs to drivers:")

# Default prompt with rules
default_prompt = """
You are a job planner system. 
Always output ONLY valid JSON, no text, no explanations. 
The JSON must follow these rules:

1. Output must be JSON only with fields: driver_id, jobs[].
2. Each job must include pickup_zone and dropoff_zone.
3. A driver's next job must have pickup_zone equal to the previous job's dropoff_zone.
4. Each driver can have at most 3 jobs.
5. Assign all jobs so none are left unassigned.

Example JSON format:
[
  {
    "driver_id": "DR-001",
    "jobs": [
      {"pickup_zone": "z1", "dropoff_zone": "z6"},
      {"pickup_zone": "z6", "dropoff_zone": "z8"}
    ]
  }
]
"""
# User input
user_prompt = st.text_area("Enter your prompt:", height=200,value=default_prompt)

# Build final prompt
final_prompt = f"""
{default_prompt}

User request:
{user_prompt}

Jobs: {json.dumps(jobs_json)}
Drivers: {json.dumps(drivers_json)}

Return the assignment strictly in JSON format, nothing else.
"""

if st.button("Run Job Assignment"):
    with st.spinner("Generating job assignments..."):
        try:
            # Run Ollama locally
            response = ollama.chat(
                model="llama3.1:8b",   # or "mistral", "llama2", etc.
                messages=[
                    {"role": "system", "content": "You are a strict JSON job assignment assistant."},
                    {"role": "user", "content": final_prompt},
                ],
                format= "json",
            )

            response_text = response["message"]["content"]
            print(response_text)
            try:
                response_json = json.loads(response_text)
                st.success("✅ Job assignment completed!")
                st.json(response_json)  # Pretty JSON
            except Exception as e:
                st.error("⚠️ Failed to parse JSON output.")
                st.text(response_text)  # Show raw output if not JSON
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

