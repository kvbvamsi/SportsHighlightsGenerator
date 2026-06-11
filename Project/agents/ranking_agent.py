# agents/ranking_agent.py

from typing import List

from models.schemas import EventData
from services.qwen_service import qwen_service


class RankingAgent:

    def __init__(self):

        self.base_scores = {
            "WICKET": 10,
            "RUNOUT": 9,
            "CATCH": 8,
            "SIX": 8,
            "FOUR": 6,
            "REVIEW": 5,
            "CELEBRATION": 4,
            "DOT_BALL": 1,
            "UNKNOWN": 0
        }

    # ------------------------------------
    # Event Type Weight
    # ------------------------------------

    def score_event_type(
        self,
        event_type: str
    ) -> float:

        return self.base_scores.get(
            event_type,
            0
        )

    # ------------------------------------
    # Crowd Excitement Weight
    # ------------------------------------

    def score_crowd(
        self,
        crowd_score: float
    ) -> float:

        return crowd_score * 5

    # ------------------------------------
    # Match Phase Weight
    # ------------------------------------

    def score_match_phase(
        self,
        overs: float
    ) -> float:

        if overs >= 18:
            return 5

        elif overs >= 15:
            return 4

        elif overs >= 10:
            return 3

        elif overs >= 5:
            return 2

        return 1

    # ------------------------------------
    # Score Pressure
    # ------------------------------------

    def score_pressure(
        self,
        scoreboard
    ) -> float:

        try:

            wickets = scoreboard.wickets

            if wickets >= 7:
                return 5

            elif wickets >= 5:
                return 3

            return 1

        except Exception:
            return 0

    # ------------------------------------
    # Extract Over Number
    # ------------------------------------

    def extract_overs(
        self,
        event: EventData
    ):

        try:

            if event.scoreboard:

                over_text = (
                    event.scoreboard.overs
                )

                return float(
                    over_text
                )

        except Exception:
            pass

        return 0

    # ------------------------------------
    # LLM Importance Score
    # ------------------------------------

    def llm_importance_score(
        self,
        event: EventData
    ):

        prompt = f"""
You are a cricket highlight analyst.

Rate importance of this event.

Event Type:
{event.event_type}

Commentary:
{event.commentary}

Return only number from 1 to 10.
"""

        try:

            score = qwen_service.generate(
                prompt
            )

            return float(
                str(score).strip()
            )

        except Exception:

            return 5

    # ------------------------------------
    # Final Score
    # ------------------------------------

    def score_event(
        self,
        event: EventData
    ):

        score = 0

        score += self.score_event_type(
            event.event_type
        )

        score += self.score_crowd(
            event.crowd_score
        )

        score += self.score_pressure(
            event.scoreboard
        )

        score += self.score_match_phase(
            self.extract_overs(
                event
            )
        )

        score += self.llm_importance_score(
            event
        )

        return round(score, 2)

    # ------------------------------------
    # Rank Events
    # ------------------------------------

    def rank_events(
        self,
        events: List[EventData]
    ):

        ranked = []

        for event in events:

            score = self.score_event(
                event
            )

            ranked.append(
                {
                    "event": event,
                    "score": score
                }
            )

        ranked.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return ranked

    # ------------------------------------
    # Top Highlights
    # ------------------------------------

    def top_highlights(
        self,
        events: List[EventData],
        top_k: int = 10
    ):

        ranked = self.rank_events(
            events
        )

        return ranked[:top_k]


ranking_agent = RankingAgent()