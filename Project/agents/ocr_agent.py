# agents/ocr_agent.py

import re
import cv2
import yaml
import torch

from typing import Optional

from PIL import Image

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel
)

from paddleocr import PaddleOCR

from models.schemas import (
    FrameData,
    OCRResponse,
    ScoreboardData
)


class OCRAgent:

    def __init__(
        self,
        config_path="configs/settings.yaml"
    ):

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        trocr_model_name = self.config["ocr"]["trocr_model"]

        print(
            f"Loading TrOCR: {trocr_model_name}"
        )

        self.processor = (
            TrOCRProcessor.from_pretrained(
                trocr_model_name
            )
        )

        self.model = (
            VisionEncoderDecoderModel.from_pretrained(
                trocr_model_name
            )
        )

        self.paddle = PaddleOCR(
            use_angle_cls=True,
            lang="en"
        )

    # ---------------------------------
    # Scoreboard Region Detection
    # ---------------------------------

    def crop_scoreboard_region(
        self,
        frame_path: str
    ):

        image = cv2.imread(frame_path)

        if image is None:
            raise ValueError(
                f"Cannot load {frame_path}"
            )

        h, w = image.shape[:2]

        # Broadcast scoreboard usually top-left

        roi = image[
            0:int(h * 0.18),
            0:int(w * 0.45)
        ]

        return roi

    # ---------------------------------
    # TrOCR
    # ---------------------------------

    def run_trocr(
        self,
        image_array
    ) -> str:

        pil_image = Image.fromarray(
            cv2.cvtColor(
                image_array,
                cv2.COLOR_BGR2RGB
            )
        )

        pixel_values = self.processor(
            pil_image,
            return_tensors="pt"
        ).pixel_values

        generated_ids = self.model.generate(
            pixel_values
        )

        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        return text

    # ---------------------------------
    # PaddleOCR Fallback
    # ---------------------------------

    def run_paddleocr(
        self,
        image_array
    ) -> str:

        result = self.paddle.ocr(
            image_array,
            cls=True
        )

        lines = []

        if result:

            for block in result:

                if block:

                    for item in block:

                        lines.append(
                            item[1][0]
                        )

        return " ".join(lines)

    # ---------------------------------
    # Parse Scoreboard
    # ---------------------------------

    def parse_scoreboard(
        self,
        text: str
    ) -> Optional[ScoreboardData]:

        score_match = re.search(
            r'(\d{1,3})\/(\d{1,2})',
            text
        )

        over_match = re.search(
            r'(\d{1,2}\.\d)',
            text
        )

        if not score_match:
            return None

        score = score_match.group(0)

        wickets = int(
            score_match.group(2)
        )

        overs = (
            over_match.group(1)
            if over_match
            else ""
        )

        return ScoreboardData(
            score=score,
            wickets=wickets,
            overs=overs
        )

    # ---------------------------------
    # Extract Scoreboard
    # ---------------------------------

    def extract_scoreboard(
        self,
        frame: FrameData
    ) -> Optional[OCRResponse]:

        roi = self.crop_scoreboard_region(
            frame.frame_path
        )

        text = self.run_trocr(
            roi
        )

        scoreboard = (
            self.parse_scoreboard(text)
        )

        if scoreboard is None:

            text = self.run_paddleocr(
                roi
            )

            scoreboard = (
                self.parse_scoreboard(text)
            )

        if scoreboard is None:
            return None

        return OCRResponse(
            timestamp=frame.timestamp,
            scoreboard=scoreboard
        )

    # ---------------------------------
    # Batch Processing
    # ---------------------------------

    def process_frames(
        self,
        frames
    ):

        results = []

        for frame in frames:

            try:

                data = self.extract_scoreboard(
                    frame
                )

                if data:
                    results.append(data)

            except Exception as e:

                print(
                    f"OCR error: {e}"
                )

        return results


ocr_agent = OCRAgent()