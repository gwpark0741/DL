from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ScoreResult:
    label: str
    score: float
    grade: str
    tip: Optional[str] = None


@dataclass
class RepEvaluation:
    rep_id: int
    view_mode: str
    scores: List[ScoreResult]
    probabilities: List[float]


class SquatView1Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(10, 64, num_layers=1, batch_first=True)
        self.layer_norm = nn.LayerNorm(64)
        self.fc = nn.Linear(64, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        h0 = torch.zeros(1, x.size(0), 64, device=x.device)
        c0 = torch.zeros(1, x.size(0), 64, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(self.layer_norm(out[:, -1, :]))


class SquatView3Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(6, 128, num_layers=2, batch_first=True)
        self.layer_norm = nn.LayerNorm(128)
        self.fc = nn.Linear(128, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        h0 = torch.zeros(2, x.size(0), 128, device=x.device)
        c0 = torch.zeros(2, x.size(0), 128, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(self.layer_norm(out[:, -1, :]))


class SquatAnalyzer:
    def __init__(self, model_v1_path: str, model_v3_path: str) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_v1 = SquatView1Model().to(self.device)
        self.model_v3 = SquatView3Model().to(self.device)

        if os.path.exists(model_v1_path):
            self.model_v1.load_state_dict(torch.load(model_v1_path, map_location=self.device))
            self.model_v1.eval()
        else:
            raise FileNotFoundError(f"View1 model not found: {model_v1_path}")

        if os.path.exists(model_v3_path):
            self.model_v3.load_state_dict(torch.load(model_v3_path, map_location=self.device))
            self.model_v3.eval()
        else:
            raise FileNotFoundError(f"View3 model not found: {model_v3_path}")

        self.v1_keys: List[str] = [
            "trunk_inclination",
            "ankle_dorsi_left",
            "ankle_dorsi_right",
            "hip_flexion_left",
            "hip_flexion_right",
            "knee_mean",
            "tibia_inclination_left",
            "tibia_inclination_right",
            "face_shoulder_hip_left",
            "face_shoulder_hip_right",
        ]
        self.v1_mean = np.array(
            [
                13.5081,
                119.9536,
                119.0998,
                156.8377,
                128.0947,
                144.0267,
                14.4445,
                9.7183,
                127.5440,
                162.1496,
            ],
            dtype=np.float32,
        )
        self.v1_std = np.array(
            [
                9.0042,
                15.7305,
                9.8618,
                23.7940,
                24.7444,
                25.4835,
                8.6527,
                8.9265,
                10.9611,
                19.4002,
            ],
            dtype=np.float32,
        )

        self.v3_keys: List[str] = [
            "fppa_left",
            "fppa_right",
            "pelvic_tilt",
            "trunk_lateral_tilt",
            "knee_width_ratio",
            "head_pitch_proxy",
        ]
        self.v3_mean = np.array(
            [166.6711, 166.5661, 178.8124, 178.4266, 0.8705, 0.5871], dtype=np.float32
        )
        self.v3_std = np.array(
            [16.2031, 16.3651, 5.5688, 4.2768, 0.2692, 0.2303], dtype=np.float32
        )

    def _preprocess_sequence(
        self, sequence: List[Dict[str, float]], keys: List[str], mean: np.ndarray, std: np.ndarray
    ) -> torch.Tensor:
        frames: List[List[float]] = []
        for frame in sequence:
            frames.append([float(frame.get(k, 0.0)) for k in keys])

        if not frames:
            raise ValueError("No frames provided to preprocess")

        if len(frames) > 64:
            frames = frames[:64]
        while len(frames) < 64:
            frames.append(frames[-1])

        X = (np.array(frames, dtype=np.float32) - mean) / std
        X = np.nan_to_num(X)
        return torch.tensor(X).unsqueeze(0).to(self.device)

    @staticmethod
    def _grade(score: float) -> str:
        if score >= 80:
            return "⭐⭐⭐ (완벽)"
        if score >= 60:
            return "⭐⭐ (양호)"
        if score >= 40:
            return "⚠️ (주의)"
        return "❌ (교정 필요)"

    @staticmethod
    def _tip(label: str) -> str:
        tips = {
            "척추 중립": "허리가 굽지 않도록 가슴을 펴고 시선을 정면에 두세요.",
            "발바닥 고정": "발 전체로 바닥을 눌러 중심을 안정시키세요.",
            "가동 범위": "무릎이 흔들리지 않는 범위에서 깊이를 조금 더 가져가세요.",
            "무릎 정렬": "무릎이 안쪽으로 말리지 않도록 발끝과 같은 방향으로 밀어주세요.",
        }
        return tips.get(label, "")

    def evaluate_rep(self, rep_id: int, rep_frames: List[Dict[str, float]], view_mode: str) -> RepEvaluation:
        if view_mode == "side":
            input_tensor = self._preprocess_sequence(rep_frames, self.v1_keys, self.v1_mean, self.v1_std)
            with torch.no_grad():
                probs = torch.sigmoid(self.model_v1(input_tensor)).cpu().numpy()[0]

            labels = ["척추 중립", "발바닥 고정", "가동 범위"]
            scores: List[ScoreResult] = []
            for label, prob in zip(labels, probs):
                score = float(100 - (prob * 100))
                grade = self._grade(score)
                tip = self._tip(label) if score < 60 else None
                scores.append(ScoreResult(label=label, score=score, grade=grade, tip=tip))
            return RepEvaluation(rep_id=rep_id, view_mode=view_mode, scores=scores, probabilities=probs.tolist())

        if view_mode == "front":
            input_tensor = self._preprocess_sequence(rep_frames, self.v3_keys, self.v3_mean, self.v3_std)
            with torch.no_grad():
                probs = torch.sigmoid(self.model_v3(input_tensor)).cpu().numpy()[0]

            knee_score = float(100 - (probs[1] * 100))
            head_score = float(100 - (probs[0] * 100))

            scores = [
                ScoreResult(
                    label="무릎 정렬",
                    score=knee_score,
                    grade=self._grade(knee_score),
                    tip=self._tip("무릎 정렬") if knee_score < 60 else None,
                ),
                ScoreResult(
                    label="시선 처리",
                    score=head_score,
                    grade=self._grade(head_score),
                    tip=None,
                ),
            ]
            return RepEvaluation(rep_id=rep_id, view_mode=view_mode, scores=scores, probabilities=probs.tolist())

        raise ValueError("view_mode must be 'side' or 'front'")
