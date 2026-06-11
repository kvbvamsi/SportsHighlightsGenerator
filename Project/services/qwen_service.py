# services/qwen_service.py

import os
import yaml
from openai import OpenAI


class QwenService:
    """
    Shared service for all ADK agents.
    Connects to vLLM OpenAI-compatible endpoint.
    """

    def __init__(self, config_path="configs/settings.yaml"):

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.client = OpenAI(
            base_url=self.config["llm"]["base_url"],
            api_key=self.config["llm"]["api_key"]
        )

        self.model_name = self.config["llm"]["model_name"]

    def generate(
        self,
        prompt: str,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:

        temperature = (
            temperature
            if temperature is not None
            else self.config["llm"]["temperature"]
        )

        max_tokens = (
            max_tokens
            if max_tokens is not None
            else self.config["llm"]["max_tokens"]
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content

    def generate_json(
        self,
        prompt: str,
        temperature: float = 0.1
    ) -> dict:

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            response_format={"type": "json_object"}
        )

        return response.choices[0].message.content

    def summarize_event(
        self,
        event_type: str,
        player: str,
        score: str
    ) -> str:

        prompt = f"""
        Generate a short cricket highlight description.

        Event: {event_type}
        Player: {player}
        Score: {score}

        Output:
        One exciting sentence.
        """

        return self.generate(prompt)

    def rank_event(
        self,
        event_text: str
    ) -> str:

        prompt = f"""
        Rate this cricket event importance from 1 to 10.

        Event:
        {event_text}

        Return only the number.
        """

        return self.generate(prompt)

    def generate_narration(
        self,
        event_text: str
    ) -> str:

        prompt = f"""
        You are a professional cricket commentator.

        Create a 2-3 sentence narration for:

        {event_text}
        """

        return self.generate(prompt)


qwen_service = QwenService()