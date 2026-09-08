import os
import subprocess
import sys

base = os.path.dirname(os.path.abspath(__file__))
main = os.path.join(base, "app.py")
os.chdir(base)

subprocess.run([
    sys.executable,
    "-m", "streamlit",
    "run", main,
    "--server.port", "8501",
])
