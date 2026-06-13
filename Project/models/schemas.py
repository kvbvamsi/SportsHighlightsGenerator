# models/schemas.py

from typing import List, Optional
from pydantic import BaseModel, Field


# =====================================================
# VIDEO FRAME SCHEMA
# =====================================================

class FrameData(BaseModel):
    """
    Stores extracted video frame information.
    """

    frame_id: int

    timestamp: float

    image_path: str

    width: Optional[int] = None

    height: Optional[int] = None



# =====================================================
# OCR OUTPUT
# =====================================================

class OCRResult(BaseModel):

    frame_id: int

    timestamp: float

    text: str



# =====================================================
# AUDIO / COMMENTARY ANALYSIS
# =====================================================

class AudioEvent(BaseModel):

    timestamp: float

    transcript: str

    excitement_score: float = 0.0



# =====================================================
# SPORTS EVENT DETECTION
# =====================================================

class EventData(BaseModel):

    event_type: str = Field(
        description=
        """
        Cricket events:
        FOUR
        SIX
        WICKET
        CATCH
        RUNOUT
        BOUNDARY
        CELEBRATION
        """
    )

    timestamp: float

    confidence: float

    score_text: Optional[str] = None

    commentary: Optional[str] = None

    frame_id: Optional[int] = None



# =====================================================
# EVENT RANKING
# =====================================================

class RankedEvent(BaseModel):

    event: EventData

    importance_score: float

    rank: int



# =====================================================
# HIGHLIGHT SEGMENT
# =====================================================

class HighlightSegment(BaseModel):

    start_time: float

    end_time: float

    event_type: str

    score: float



# =====================================================
# FINAL OUTPUT
# =====================================================

class HighlightOutput(BaseModel):

    video_path: str

    highlights: List[HighlightSegment]

    total_events: int

    generated_at: str



# =====================================================
# AGENT RESPONSE
# =====================================================

class AgentResponse(BaseModel):

    success: bool

    message: str

    data: Optional[dict] = None



# =====================================================
# MCP MESSAGE FORMAT
# =====================================================

class MCPRequest(BaseModel):

    query: str

    context: Optional[dict] = None



class MCPResponse(BaseModel):

    answer: str

    confidence: float

    metadata: Optional[dict] = None