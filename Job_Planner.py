import streamlit as st
import json
import paramiko
import os 
from dotenv import load_dotenv
import requests

load_dotenv()

# -------------------------
# Load JSON files on Windows
# -------------------------
with open("C:\\Job Planner\\Json\\drivers.json", "r") as f:
    drivers_json = json.load(f)

with open("C:\\Job Planner\\Json\\jobs.json", "r") as f:
    jobs_json = json.load(f)

# Streamlit UI
st.write("Enter your prompt below to assign jobs to drivers:")

# Default prompt
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
            # -------------------------
            # Connect to Mac via SSH
            # -------------------------
            MAC_IP = "192.168.1.19"   # replace with your Mac IP
            USERNAME = "darshal"
            PASSWORD = "dm@mac"  # or use key-based auth
            # PASSWORD = os.getenv("PASSWORD")
            # print(PASSWORD)
            OLLAMA_PATH = "~/Code/Job Planner/ollama/ollama"

            KEY_PATH="C:\\Users\\Admin\\.ssh\\id_rsa"    

            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            # ssh.connect(MAC_IP, username=USERNAME, password=PASSWORD)

            try:
                # ✅ Try SSH key first
                if os.path.exists(KEY_PATH):
                    ssh.connect(MAC_IP, username=USERNAME, key_filename=KEY_PATH, timeout=10)
                else:
                    raise FileNotFoundError("SSH key not found")
            except Exception:
                # 🔑 Fallback to password
                ssh.connect(MAC_IP, username=USERNAME, password=PASSWORD, timeout=10)


            # -------------------------
            # Run Ollama command and send prompt via stdin
            # -------------------------

            command = f'echo "{final_prompt}" | {OLLAMA_PATH} run llama3.1:8b --input -'
            stdin, stdout, stderr = ssh.exec_command(command)

            # Send final_prompt as input to Ollama
            stdin.write(final_prompt)
            stdin.channel.shutdown_write()


            response_text = stdout.read().decode().strip()
            error_text = stderr.read().decode().strip()

            ssh.close()

            # -------------------------
            # Parse output
            # -------------------------
            try:
                response_json = json.loads(response_text)
                st.success("✅ Job assignment completed!")
                st.json(response_json)
            except Exception:
                st.error("⚠️ Failed to parse JSON output.")
                st.text(response_text)
                if error_text:
                    st.text("Errors from Ollama: " + error_text)

        except Exception as e:
            st.error(f"❌ SSH/Ollama error: {str(e)}")

