# main.py

import os
import argparse
import logging

from agents.video_agent import video_agent
from agents.audio_agent import audio_agent
from agents.ocr_agent import ocr_agent
from agents.event_agent import event_agent
from agents.ranking_agent import ranking_agent
from agents.highlight_agent import highlight_agent

from services.video_editor import video_editor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class HighlightPipeline:

    def __init__(self):

        pass

    # -----------------------------------
    # Step 1
    # -----------------------------------

    def extract_frames(
        self,
        video_path
    ):

        logger.info(
            "Extracting frames..."
        )

        frames = video_agent.extract_frames(
            video_path
        )

        logger.info(
            f"{len(frames)} frames extracted"
        )

        return frames

    # -----------------------------------
    # Step 2
    # -----------------------------------

    def run_ocr(
        self,
        frames
    ):

        logger.info(
            "Running OCR..."
        )

        ocr_results = (
            ocr_agent.process_frames(
                frames
            )
        )

        logger.info(
            f"{len(ocr_results)} OCR results"
        )

        return ocr_results

    # -----------------------------------
    # Step 3
    # -----------------------------------

    def process_audio(
        self,
        video_path
    ):

        logger.info(
            "Processing audio..."
        )

        audio_results = (
            audio_agent.analyze_audio(
                video_path
            )
        )

        logger.info(
            f"{len(audio_results)} audio events"
        )

        return audio_results

    # -----------------------------------
    # Step 4
    # -----------------------------------

    def detect_events(
        self,
        audio_results,
        ocr_results
    ):

        logger.info(
            "Detecting events..."
        )

        events = (
            event_agent.detect_events(
                audio_results,
                ocr_results
            )
        )

        logger.info(
            f"{len(events)} events detected"
        )

        return events

    # -----------------------------------
    # Step 5
    # -----------------------------------

    def rank_events(
        self,
        events
    ):

        logger.info(
            "Ranking events..."
        )

        ranked_events = (
            ranking_agent.rank_events(
                events
            )
        )

        logger.info(
            f"{len(ranked_events)} events ranked"
        )

        return ranked_events

    # -----------------------------------
    # Step 6
    # -----------------------------------

    def select_highlights(
        self,
        ranked_events,
        top_k
    ):

        logger.info(
            "Selecting highlights..."
        )

        selected = (
            ranking_agent.top_highlights(
                [r["event"] for r in ranked_events],
                top_k=top_k
            )
        )

        return selected

    # -----------------------------------
    # Step 7
    # -----------------------------------

    def build_segments(
        self,
        ranked_events
    ):

        logger.info(
            "Building segments..."
        )

        segments = (
            highlight_agent.build_highlights(
                ranked_events
            )
        )

        logger.info(
            f"{len(segments)} segments created"
        )

        return segments

    # -----------------------------------
    # Step 8
    # -----------------------------------

    def create_reel(
        self,
        video_path,
        segments,
        output_video
    ):

        logger.info(
            "Generating highlight reel..."
        )

        output = (
            video_editor.create_highlight_reel(
                input_video=video_path,
                segments=segments,
                output_video=output_video
            )
        )

        logger.info(
            f"Saved: {output}"
        )

        return output

    # -----------------------------------
    # Full Pipeline
    # -----------------------------------

    def run(
        self,
        video_path,
        output_video,
        top_k
    ):

        logger.info(
            "=" * 60
        )

        logger.info(
            "STARTING HIGHLIGHT PIPELINE"
        )

        logger.info(
            "=" * 60
        )

        frames = self.extract_frames(
            video_path
        )

        ocr_results = self.run_ocr(
            frames
        )

        audio_results = self.process_audio(
            video_path
        )

        events = self.detect_events(
            audio_results,
            ocr_results
        )

        ranked_events = self.rank_events(
            events
        )

        segments = self.build_segments(
            ranked_events[:top_k]
        )

        final_video = self.create_reel(
            video_path,
            segments,
            output_video
        )

        logger.info(
            "=" * 60
        )

        logger.info(
            "PIPELINE COMPLETED"
        )

        logger.info(
            "=" * 60
        )

        return final_video


# ---------------------------------------
# CLI
# ---------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        required=True,
        help="Input cricket video"
    )

    parser.add_argument(
        "--output",
        default="data/output/highlights.mp4"
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=10
    )

    args = parser.parse_args()

    pipeline = HighlightPipeline()

    pipeline.run(
        video_path=args.video,
        output_video=args.output,
        top_k=args.top_k
    )


if __name__ == "__main__":

    main()