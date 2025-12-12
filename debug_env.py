import sys
import os

print("Python Executable:", sys.executable)
print("System Path:")
for p in sys.path:
    print(p)

try:
    import ollama
    print("Ollama imported successfully from:", ollama.__file__)
except ImportError as e:
    print("Failed to import ollama:", e)

try:
    import langchain_ollama
    print("Langchain Ollama imported successfully from:", langchain_ollama.__file__)
except ImportError as e:
    print("Failed to import langchain_ollama:", e)
