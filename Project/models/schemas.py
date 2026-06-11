# models/schemas.py

from typing import List, Optional
from pydantic import BaseModel, Field


# -----------------------------------------
# OCR
# -----------------------------------------

class OCRResult(BaseModel):
    timestamp: float
    text: str


# -----------------------------------------
# Audio
# -----------------------------------------

class AudioEvent(BaseModel):
    timestamp: float
    transcript: str
    excitement_score: float = 0.0


# -----------------------------------------
# Event Detection
# -----------------------------------------

class EventData(BaseModel):

    event_type: str = Field(
        description="FOUR, SIX, WICKET, GOAL, SHOT, SAVE"
    )

    timestamp: float

    confidence: float

    score_text: Optional[str] = None

    commentary: Optional[str] = None


# -----------------------------------------
# Ranked Event
# -----------------------------------------

class RankedEvent(BaseModel):

    event: EventData

    importance_score: float

    rank: int


# -----------------------------------------
# Highlight Segment
# -----------------------------------------

class HighlightSegment(BaseModel):

    start_time: float

    end_time: float

    event_type: str

    score: float


# -----------------------------------------
# Final Output
# -----------------------------------------

class HighlightOutput(BaseModel):

    video_path: str

    highlights: List[HighlightSegment]

    total_events: int

    generated_at: str


# -----------------------------------------
# Agent Response
# -----------------------------------------

class AgentResponse(BaseModel):

    success: bool

    message: str

    data: Optional[dict] = None


# -----------------------------------------
# MCP Request
# -----------------------------------------

class MCPRequest(BaseModel):

    query: str

    context: Optional[dict] = None


# -----------------------------------------
# MCP Response
# -----------------------------------------

class MCPResponse(BaseModel):

    answer: str

    confidence: float

    metadata: Optional[dict] = None