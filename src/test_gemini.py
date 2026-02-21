import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load your API key from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=api_key)

# Choose the model - updated model name
model = genai.GenerativeModel("gemini-2.5-flash-lite")

# Send a message
response = model.generate_content("What are the top 3 things a beginner should know about personal finance in India?")

# Print the response
print(response.text)