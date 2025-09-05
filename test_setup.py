import os
from dotenv import load_dotenv
import openai

#load environment variables
load_dotenv()

#check if API is loaded
api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    print("envionment variables loaded successfully")
    print(f"API key found")
else:
    print("API not found..check .env file")

#test imports
try:
    import langchain
    print("Langchain import sucess")
except ImportError as e:
    print(f"failed:{e}")

try:
    import openai
    print("Openai import sucess")
except ImportError as e:
    print(f"failed:{e}")

