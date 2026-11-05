from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if api_key:
    print("GROQ API key loaded successfully!")
else:
    print("GROQ API key not found!")