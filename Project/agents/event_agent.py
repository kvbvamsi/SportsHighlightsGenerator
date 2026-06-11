
# agents/event_agent.py

import json
import uuid
from typing import List, Optional

from models.schemas import (
    AudioData,
    OCRResponse,
    EventData
)

from services.qwen_service import qwen_service


EVENT_TYPES = [
    "SIX",
    "FOUR",
    "WICKET",
    "CATCH",
    "RUNOUT",
    "REVIEW",
    "CELEBRATION",
    "DOT_BALL",
    "UNKNOWN"
]


class EventAgent:

    def __init__(self):

        self.keyword_map = {
            "SIX": [
                "six",
                "maximum",
                "huge hit",'into stands','into crouds','air'
            ],

            "FOUR": [
                "four",
                "boundary"
            ],

            "WICKET": [
                "out",
                "bowled",
                "caught",
                "lbw",
                "cleaned up"
            ],

            "RUNOUT": [
                "run out",
                "direct hit"
            ],

            "REVIEW": [
                "review",
                "drs"
            ]
        }

    # -----------------------------------------
    # Fast Rule Based Detection
    # -----------------------------------------

    def detect_from_commentary(
        self,
        commentary: str
    ):

        text = commentary.lower()

        for event_type, keywords in self.keyword_map.items():

            for keyword in keywords:

                if keyword in text:

                    return (
                        event_type,
                        0.80
                    )

        return (
            "UNKNOWN",
            0.30
        )

    # -----------------------------------------
    # Qwen Reasoning Layer
    # -----------------------------------------

    def qwen_classify_event(
        self,
        commentary: str
    ):

        prompt = f"""
You are a cricket analyst.

Classify the event.

Commentary:
{commentary}

Possible Events:
{EVENT_TYPES}

Return JSON only.

Example:
{{
  "event_type":"SIX",
  "confidence":0.95
}}
"""

        try:

            result = qwen_service.generate_json(
                prompt
            )

            if isinstance(result, str):
                result = json.loads(result)

            return (
                result["event_type"],
                float(result["confidence"])
            )

        except Exception:

            return (
                "UNKNOWN",
                0.20
            )

    # -----------------------------------------
    # Match OCR With Audio
    # -----------------------------------------

    def nearest_scoreboard(
        self,
        timestamp: float,
        scoreboards: List[OCRResponse]
    ):

        if len(scoreboards) == 0:
            return None

        nearest = min(
            scoreboards,
            key=lambda x: abs(
                x.timestamp - timestamp
            )
        )

        return nearest.scoreboard

    # -----------------------------------------
    # Event Detection
    # -----------------------------------------

    def detect_events(
        self,
        audio_results: List[AudioData],
        ocr_results: List[OCRResponse]
    ) -> List[EventData]:

        events = []

        for audio in audio_results:

            event_type, confidence = (
                self.detect_from_commentary(
                    audio.commentary_text
                )
            )

            if event_type == "UNKNOWN":

                event_type, confidence = (
                    self.qwen_classify_event(
                        audio.commentary_text
                    )
                )

            if event_type == "UNKNOWN":
                continue

            scoreboard = self.nearest_scoreboard(
                audio.timestamp,
                ocr_results
            )

            event = EventData(
                event_id=str(
                    uuid.uuid4()
                ),

                timestamp=audio.timestamp,

                event_type=event_type,

                confidence=confidence,

                scoreboard=scoreboard,

                commentary=audio.commentary_text,

                crowd_score=audio.crowd_intensity
            )

            events.append(
                event
            )

        return events

    # -----------------------------------------
    # High Impact Filter
    # -----------------------------------------

    def filter_major_events(
        self,
        events: List[EventData]
    ):

        major = []

        important_events = [
            "SIX",
            "FOUR",
            "WICKET",
            "RUNOUT",
            "CATCH"
        ]

        for event in events:

            if (
                event.event_type
                in important_events
            ):
                major.append(
                    event
                )

        return major


event_agent = EventAgent()