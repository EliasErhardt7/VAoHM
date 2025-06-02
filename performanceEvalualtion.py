from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np

try:
    from fastdtw import fastdtw  # type: ignore
    from scipy.spatial.distance import euclidean  # type: ignore

    _HAS_FDTW = True
except ImportError:  # graceful degradation

    def fastdtw(a, b, dist):  # fall‑back using NumPy only (slow!)
        from math import inf

        n, m = len(a), len(b)
        dtw = np.full((n + 1, m + 1), inf, dtype=float)
        dtw[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = dist(a[i - 1], b[j - 1])
                dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
        return dtw[n, m], None

    from numpy.linalg import norm as euclidean  # type: ignore

    _HAS_FDTW = False

    
# 1.  Per‑repetition sequence recording                                       


class RepRecorder:
    """Accumulate per‑frame feature vectors and segment them into
    individual repetitions using the *state* signal from
    ``ExerciseCounter``.

    Parameters
    ----------
    history : int, optional
        How many frames to keep in memory for the *current* repetition
        (helps avoid unbounded RAM use on extremely long videos).
    """

    def __init__(self, history: int = 1024):
        self._current: List[np.ndarray] = []
        self.sequences: List[List[np.ndarray]] = []
        self._in_rep = False
        self._history = history

    def update(self, feature_vec: np.ndarray, state: str | None) -> None:
        if state == "down":
            # New repetition starts → (re‑)initialise buffer
            if not self._in_rep:
                self._current = []
                self._in_rep = True
            self._append(feature_vec)
        elif state == "up":
            if self._in_rep:
                self._append(feature_vec)
                # Repetition finished → store and clear
                self.sequences.append(self._current)
                self._current = []
                self._in_rep = False
        else:  # state == "in motion" or None
            if self._in_rep:
                self._append(feature_vec)


    def pop_last(self) -> List[np.ndarray] | None:
        """Return the most recently finished repetition (or ``None``)."""
        return self.sequences.pop() if self.sequences else None


    def _append(self, vec: np.ndarray) -> None:
        self._current.append(vec)
        if len(self._current) > self._history:
            self._current.pop(0)  # softly constrain memory


# 2.  Comparing a repetition against the ideal template                      


class PerformanceEvaluator:
    """Compares a repetition to a *template* and returns a 0‑100 quality
    score (100 = perfect match).

    The default transformation from DTW distance → score works
    reasonably across different movement speeds but you *should* tune
    ``alpha`` to your own data.
    """

    def __init__(self, exercise_type: str, template_dir: str | Path = "templates", *, alpha: float = 0.01):
        self.exercise = exercise_type
        self.template = self._load_template(template_dir, exercise_type)
        self.alpha = alpha

    def evaluate(self, seq: List[np.ndarray] | np.ndarray) -> float:
        if seq is None or len(seq) == 0:
            return 0.0  # malformed input → worst score
        rep = np.asarray(seq, dtype=float)
        dist, _ = fastdtw(rep, self.template, dist=euclidean)
        norm_dist = dist / len(self.template)
        score = 100.0 * np.exp(-self.alpha * norm_dist)
        return round(float(np.clip(score, 0, 100)), 1)

    @staticmethod
    def save_template(seq: List[np.ndarray] | np.ndarray, exercise: str, template_dir: str | Path = "templates") -> None:
        template_dir = Path(template_dir)
        template_dir.mkdir(parents=True, exist_ok=True)
        np.save(template_dir / f"{exercise}_template.npy", np.asarray(seq, dtype=float))

    @staticmethod
    def _load_template(template_dir: str | Path, exercise: str) -> np.ndarray:
        path = Path(template_dir) / f"{exercise}_template.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"Template file '{path}' not found. Generate it with save_template(...) first!")
        return np.load(path)


# 3.  Helper:  quick template extractor from a single ideal video         


def extract_template_from_video(video_path: str | os.PathLike, exercise: str, *,
                                feature_extractor,  # callable(frame_landmarks) -> np.ndarray
                                down_thresh: float, up_thresh: float,
                                template_dir: str | Path = "templates") -> None:
    """Run pose estimation on *video_path* and save the first repetition as
    the template for *exercise*.

    The ``feature_extractor`` is the same function you already use in
    poseDetection.py (it should return the 3‑dimensional feature vector
    [mean_elbow_angle, mean_knee_angle, mean_hip_angle]).
    """
    import cv2
    import mediapipe as mp

    from collections import deque

    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(str(video_path))
    state = None
    rep = []
    queues = deque(maxlen=5)  # quick smoothing of raw state changes

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            if not results.pose_landmarks:
                continue
            feats = feature_extractor(results.pose_landmarks.landmark)
            # Determine current phase from knee/hip/elbow angle
            angle_primary = feats[0] if exercise == "pushup" else feats[1] if exercise == "squat" else feats[2]
            current_state = "down" if angle_primary < down_thresh else "up" if angle_primary > up_thresh else "move"
            queues.append(current_state)
            majority_state = max(set(queues), key=list(queues).count)
            if majority_state == "down":
                rep.append(feats)
            elif majority_state == "up" and rep:
                # repetition finished ‑ save and break (use first rep only)
                PerformanceEvaluator.save_template(rep, exercise, template_dir)
                break
    cap.release()
