# agents/highlight_agent.py

from typing import List

from models.schemas import EventData


class HighlightAgent:

    def __init__(self):

        self.pre_event_buffer = 8
        self.post_event_buffer = 6

        self.merge_gap_seconds = 15

    # ----------------------------------
    # Single Clip
    # ----------------------------------

    def create_clip_window(
        self,
        event: EventData
    ):

        start_time = max(
            0,
            event.timestamp - self.pre_event_buffer
        )

        end_time = (
            event.timestamp +
            self.post_event_buffer
        )

        return {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "start": start_time,
            "end": end_time,
            "score": getattr(
                event,
                "ranking_score",
                0
            )
        }

    # ----------------------------------
    # Convert Events → Segments
    # ----------------------------------

    def events_to_segments(
        self,
        ranked_events
    ):

        segments = []

        for item in ranked_events:

            event = item["event"]

            event.ranking_score = item["score"]

            segments.append(
                self.create_clip_window(
                    event
                )
            )

        return segments

    # ----------------------------------
    # Merge Overlapping Clips
    # ----------------------------------

    def merge_segments(
        self,
        segments
    ):

        if len(segments) == 0:
            return []

        segments = sorted(
            segments,
            key=lambda x: x["start"]
        )

        merged = [
            segments[0]
        ]

        for current in segments[1:]:

            previous = merged[-1]

            gap = (
                current["start"]
                -
                previous["end"]
            )

            if gap <= self.merge_gap_seconds:

                previous["end"] = max(
                    previous["end"],
                    current["end"]
                )

                previous["score"] = max(
                    previous["score"],
                    current["score"]
                )

            else:

                merged.append(
                    current
                )

        return merged

    # ----------------------------------
    # Remove Tiny Segments
    # ----------------------------------

    def filter_short_segments(
        self,
        segments,
        min_duration=5
    ):

        filtered = []

        for segment in segments:

            duration = (
                segment["end"]
                -
                segment["start"]
            )

            if duration >= min_duration:

                filtered.append(
                    segment
                )

        return filtered

    # ----------------------------------
    # Main Pipeline
    # ----------------------------------

    def build_highlights(
        self,
        ranked_events
    ):

        segments = self.events_to_segments(
            ranked_events
        )

        segments = self.merge_segments(
            segments
        )

        segments = self.filter_short_segments(
            segments
        )

        return segments


highlight_agent = HighlightAgent()