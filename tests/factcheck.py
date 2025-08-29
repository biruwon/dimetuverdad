import os
from openai import OpenAI

# ✅ Make sure Ollama server is running locally: ollama serve
# and gpt-oss:20b is pulled

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def fact_check_claim(claim: str):
    """
    Uses gpt-oss-20b via Ollama to check if a claim is misinformation or biased.
    """
    prompt = f"""
    You are a fact-checking assistant.
    Task: Verify the following social media post using current web data.
    Post: "{claim}"

    Steps:
    1. Search the web for the claim.
    2. Provide a verdict: True / False / Unclear.
    3. Justify with reliable sources.
    """

    response = client.chat.completions.create(
        model="gpt-oss:20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    example_claim = "The Spanish government has announced a nationwide ban on diesel cars starting in 2025."
    print(f"Checking claim: {example_claim}\n")
    result = fact_check_claim(example_claim)
    print("Fact-check result:\n", result)
