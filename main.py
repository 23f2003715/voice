```python
import os
import json

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()


class RequestBody(BaseModel):
    document_id: str
    text: str
    schema: dict


@app.post("/")
def extract(req: RequestBody):

    prompt = f"""
Extract the invoice information.

Return ONLY valid JSON matching this schema exactly.

Schema:
{json.dumps(req.schema)}

Invoice:
{req.text}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        ),
    )

    return json.loads(response.text)
```
