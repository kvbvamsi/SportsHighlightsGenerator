# services/video_editor.py

import os
import subprocess
from pathlib import Path


class VideoEditor:

    def __init__(self):

        self.temp_dir = "data/temp"

        os.makedirs(
            self.temp_dir,
            exist_ok=True
        )

    # ----------------------------------
    # Execute FFmpeg Command
    # ----------------------------------

    def run_command(
        self,
        command
    ):

        subprocess.run(
            command,
            check=True
        )

    # ----------------------------------
    # Cut Single Clip
    # ----------------------------------

    def cut_clip(
        self,
        input_video,
        start_time,
        end_time,
        output_file
    ):

        duration = (
            end_time -
            start_time
        )

        command = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_time),
            "-i",
            input_video,
            "-t",
            str(duration),
            "-c",
            "copy",
            output_file
        ]

        self.run_command(
            command
        )

        return output_file

    # ----------------------------------
    # Create Clips
    # ----------------------------------

    def generate_clips(
        self,
        input_video,
        segments
    ):

        clips = []

        for idx, segment in enumerate(
            segments
        ):

            clip_path = os.path.join(
                self.temp_dir,
                f"clip_{idx}.mp4"
            )

            self.cut_clip(
                input_video=input_video,
                start_time=segment["start"],
                end_time=segment["end"],
                output_file=clip_path
            )

            clips.append(
                clip_path
            )

        return clips

    # ----------------------------------
    # Overlay Event Title
    # ----------------------------------

    def add_text_overlay(
        self,
        input_video,
        output_video,
        text
    ):

        command = [
            "ffmpeg",
            "-y",
            "-i",
            input_video,

            "-vf",
            (
                f"drawtext="
                f"text='{text}':"
                f"x=(w-text_w)/2:"
                f"y=50:"
                f"fontsize=40:"
                f"fontcolor=white"
            ),

            output_video
        ]

        self.run_command(
            command
        )

        return output_video

    # ----------------------------------
    # Merge Clips
    # ----------------------------------

    def merge_clips(
        self,
        clips,
        output_file
    ):

        concat_file = os.path.join(
            self.temp_dir,
            "concat.txt"
        )

        with open(
            concat_file,
            "w"
        ) as f:

            for clip in clips:

                f.write(
                    f"file '{os.path.abspath(clip)}'\n"
                )

        command = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_file,
            "-c",
            "copy",
            output_file
        ]

        self.run_command(
            command
        )

        return output_file

    # ----------------------------------
    # Generate Final Reel
    # ----------------------------------

    def create_highlight_reel(
        self,
        input_video,
        segments,
        output_video
    ):

        clips = self.generate_clips(
            input_video,
            segments
        )

        final_video = self.merge_clips(
            clips,
            output_video
        )

        return final_video

    # ----------------------------------
    # Cleanup
    # ----------------------------------

    def cleanup(self):

        temp_path = Path(
            self.temp_dir
        )

        for file in temp_path.glob("*"):

            try:
                file.unlink()

            except Exception:
                pass


video_editor = VideoEditor()