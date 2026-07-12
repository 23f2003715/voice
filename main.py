import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": "You are an invoice extraction engine. Output only JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "invoice",
                "schema": req.schema,
                "strict": True
            }
        }
    )

    return json.loads(response.choices[0].message.content)