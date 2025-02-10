import numpy as np
import os
import random
import base64
from openai import OpenAI

random.seed(123)  # Set random seed to 123

class Qwen:
    def __init__(
            self,
            api_key,
            base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
            model: str = "qwen-vl-max-latest",
        ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
    
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def query(self, image_path, prompt):
        base64_image = self.encode_image(image_path)
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return completion.choices[0].message.content

class GPT4V:
    def __init__(
            self,
            api_key,
            model: str = "gpt-4o-mini",
        ):
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
    
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def query(self, image_path, prompt):
        base64_image = self.encode_image(image_path)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
        )
        return response.choices[0]


