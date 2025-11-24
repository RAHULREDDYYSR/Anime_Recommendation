import os
from ollama import Client
from dotenv import load_dotenv
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables from .env if present
load_dotenv()
api_key = os.getenv("OLLAMA_API_KEY")

if api_key:
    client = Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {api_key}"}
    )
else:
    client = Client(host="https://ollama.com")

messages = [
    {
        "role": "user",
        "content": "write a python class to set up simple neural network"
    }
]

for part in client.chat("gpt-oss:120b", messages=messages, stream=True):
    print(part["message"]["content"], end="", flush=True)