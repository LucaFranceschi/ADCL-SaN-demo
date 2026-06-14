import os, shutil, uuid

import numpy as np
from typing import TypedDict

from .constants import *

# ======================================= SESSION MANAGEMENT =======================================

# required fields. PEP 655 unavailable
class _SessionState(TypedDict):
    session_id: str
    session_dir: str
    comparison_segs: list[np.ndarray]
    comparison_models: list[str]

class SessionState(_SessionState, total=False):
    """Per-session state dictionary"""
    image_seg: np.ndarray
    image_resolution: tuple[int, int]
    video_seg: np.ndarray
    video_resolution: tuple[int, int]
    video_audio_path: str
    video_fps: int
    comparison_resolution: tuple[int, int]


def create_session() -> SessionState:
    """Create a new session with its own directory"""
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(MEDIA_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    print(f'Created session: {session_id} at {session_dir}')

    return SessionState(
        session_id=session_id,
        session_dir=session_dir,
        comparison_segs=[],
        comparison_models=[]
    )


def cleanup_session(state: SessionState) -> None:
    """Clean up session directory and files"""
    if 'session_dir' not in state:
        return

    session_dir = state['session_dir']
    if os.path.exists(session_dir):
        shutil.rmtree(session_dir)
        print(f'Cleaned up session directory: {session_dir}')