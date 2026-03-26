from dotenv import load_dotenv
import os
import anthropic

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment
api_key = os.getenv("ANTHROPIC_API_KEY")

# Create a client — this is the object that talks to the API
client = anthropic.Anthropic(api_key=api_key)

# Send a message and get a response
response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=100,
    messages=[
        {"role": "user", "content": "Say hello in exactly one sentence."}
    ]
)

# Print the response text
print(response.content[0].text)
