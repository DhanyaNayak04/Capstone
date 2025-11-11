import requests
import os

def gemini_conversation(user_text, api_key=None, model="gemini-2.0-flash"):
    """Send user_text to Gemini API and return the response text."""
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Gemini API key not provided.")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key=" + api_key
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": user_text}]}]
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        try:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            return "Sorry, I couldn't understand the response from Gemini."
    else:
        return f"Error: {response.status_code} - {response.text}"

if __name__ == "__main__":
    api_key = "AIzaSyC1yvPtTKHY3EQ23XCB06lZQEEPdR51euE"
    while True:
        user = input("You: ")
        if user.lower() in ["exit", "quit"]:
            break
        reply = gemini_conversation(user, api_key=api_key)
        print("Gemini:", reply)
