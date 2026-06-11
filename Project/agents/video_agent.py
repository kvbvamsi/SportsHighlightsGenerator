# agents/video_agent.py

import os
import cv2
import yaml
from pathlib import Path
from typing import List

from models.schemas import FrameData


class VideoAgent:

    def __init__(self, config_path="configs/settings.yaml"):

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.fps_sampling = self.config["video"]["fps_sampling"]
        self.frames_dir = self.config["storage"]["frames_dir"]

        Path(self.frames_dir).mkdir(
            parents=True,
            exist_ok=True
        )

    def extract_frames(
        self,
        video_path: str
    ) -> List[FrameData]:

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise Exception(
                f"Unable to open video: {video_path}"
            )

        original_fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        frame_interval = int(
            max(1, original_fps * self.fps_sampling)
        )

        frames = []

        frame_count = 0
        saved_count = 0

        video_name = Path(video_path).stem

        while True:

            success, frame = cap.read()

            if not success:
                break

            if frame_count % frame_interval == 0:

                timestamp = (
                    frame_count / original_fps
                )

                frame_file = os.path.join(
                    self.frames_dir,
                    f"{video_name}_{saved_count}.jpg"
                )

                cv2.imwrite(
                    frame_file,
                    frame
                )

                frames.append(
                    FrameData(
                        frame_id=saved_count,
                        timestamp=timestamp,
                        frame_path=frame_file
                    )
                )

                saved_count += 1

            frame_count += 1

        cap.release()

        return frames

    def get_video_metadata(
        self,
        video_path: str
    ):

        cap = cv2.VideoCapture(video_path)

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        total_frames = cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )

        duration = total_frames / fps

        width = cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )

        height = cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )

        cap.release()

        return {
            "fps": fps,
            "total_frames": total_frames,
            "duration_seconds": duration,
            "resolution": f"{int(width)}x{int(height)}"
        }

    def process_video(
        self,
        video_path: str
    ):

        metadata = self.get_video_metadata(
            video_path
        )

        frames = self.extract_frames(
            video_path
        )

        return {
            "metadata": metadata,
            "frames": frames
        }


video_agent = VideoAgent()