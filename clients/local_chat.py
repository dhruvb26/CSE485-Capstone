import logging
import requests
import json

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class LocalChat:
    def __init__(self, model: str, base_url: str) -> None:
        self.model: str = model
        self.base_url: str = base_url

    def chat(self, instructions: str, messages: list[dict], temperature = 0.00, top_p = 1.00, all_negotiation_log = []) -> str:
        """Send a query using Chat Completions and return assistant content."""

        #print(f'-------------------\n{messages}\n--------------------------------')
        try:
            #temp code for debugging
            json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": instructions},
                        *[
                            {"role": message["role"], "content": message["content"]}
                            for message in messages
                        ],
                    ],
                    "max_tokens": 1024,
                    "temperature": temperature,
                    "top_p": top_p,
                    "echo": True
                }
            r = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Content-Type": "application/json"},
                json=json,
                timeout=120,
            )
            if r.status_code != 200:
                print("Status:", r.status_code)
                print("Response:", r.text)
                r.raise_for_status()
            data = r.json()
            #for input output extraction and debugging
            all_negotiation_log.append((json, data))
            
            #used for debugging
            print(data.get("prompt", "NO PROMPT FIELD RETURNED"))

            return (data["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:
            logger.info("Error sending query to LocalChat. Returning empty string.")
            print(f"Error in LocalChat: {e}")
            return ""
        return ""

def test_local():
    model = LocalChat("mistralai/Mistral-7B-Instruct-v0.2", "http://127.0.0.1:8000/v1")
    out = model.chat("You are a cat", [{"role": "user", "content": "Bark"}])
    print(out)

    url = "http://127.0.0.1:8000/v1/chat/completions"

    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain how transformers work in simple terms."}
        ],
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    print(response.json())

