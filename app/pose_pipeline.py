from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

KEYPOINTS = [
    "Nose",
    "Left Eye",
    "Right Eye",
    "Left Shoulder",
    "Right Shoulder",
    "Left Hip",
    "Right Hip",
    "Left Knee",
    "Right Knee",
    "Left Ankle",
    "Right Ankle",
    "Left Foot",
    "Right Foot",
]

MP_MAP = {
    "Nose": mp_pose.PoseLandmark.NOSE,
    "Left Eye": mp_pose.PoseLandmark.LEFT_EYE,
    "Right Eye": mp_pose.PoseLandmark.RIGHT_EYE,
    "Left Shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "Right Shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "Left Hip": mp_pose.PoseLandmark.LEFT_HIP,
    "Right Hip": mp_pose.PoseLandmark.RIGHT_HIP,
    "Left Knee": mp_pose.PoseLandmark.LEFT_KNEE,
    "Right Knee": mp_pose.PoseLandmark.RIGHT_KNEE,
    "Left Ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
    "Right Ankle": mp_pose.PoseLandmark.RIGHT_ANKLE,
    "Left Foot": mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
    "Right Foot": mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
}

SIDE_VIEW_COLUMNS = [
    "knee_left",
    "knee_right",
    "knee_mean",
    "hip_flexion_left",
    "hip_flexion_right",
    "hip_flexion_mean",
    "ankle_dorsi_left",
    "ankle_dorsi_right",
    "ankle_dorsi_mean",
    "trunk_inclination",
    "tibia_inclination_left",
    "tibia_inclination_right",
    "tibia_inclination_mean",
    "face_shoulder_hip_left",
    "face_shoulder_hip_right",
]

FRONT_VIEW_COLUMNS = [
    "fppa_left",
    "fppa_right",
    "fppa_mean",
    "pelvic_tilt",
    "trunk_lateral_tilt",
    "knee_width_ratio",
    "head_pitch_proxy",
]


# ---------- 기하학/각도 계산 유틸 ----------

def get_pt(pts: Dict[str, Dict[str, float]], name: str) -> np.ndarray | None:
    point = pts.get(name)
    if point is None:
        return None
    return np.array([point["x"], point["y"]], dtype=float)


def l2(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def angle_between(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-8) -> float | None:
    n1, n2 = l2(v1), l2(v2)
    if n1 < eps or n2 < eps:
        return None
    cosine = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def joint_angle(A: np.ndarray | None, B: np.ndarray | None, C: np.ndarray | None) -> float | None:
    if A is None or B is None or C is None:
        return None
    v1 = A - B
    v2 = C - B
    return angle_between(v1, v2)


def midpoint(P: np.ndarray | None, Q: np.ndarray | None) -> np.ndarray | None:
    if P is None or Q is None:
        return None
    return (P + Q) / 2


def dist(P: np.ndarray | None, Q: np.ndarray | None) -> float | None:
    if P is None or Q is None:
        return None
    return float(np.linalg.norm(P - Q))


V_UP = np.array([0.0, -1.0], dtype=float)
H_RIGHT = np.array([1.0, 0.0], dtype=float)


def compute_all_angles(pts: Dict[str, Dict[str, float]], existing_angles: Dict[str, float] | None = None) -> Dict[str, float]:
    ang: Dict[str, float] = dict(existing_angles) if existing_angles else {}

    Nose = get_pt(pts, "Nose")
    LEye = get_pt(pts, "Left Eye")
    REye = get_pt(pts, "Right Eye")

    LS = get_pt(pts, "Left Shoulder")
    RS = get_pt(pts, "Right Shoulder")
    LH = get_pt(pts, "Left Hip")
    RH = get_pt(pts, "Right Hip")
    LK = get_pt(pts, "Left Knee")
    RK = get_pt(pts, "Right Knee")
    LA = get_pt(pts, "Left Ankle")
    RA = get_pt(pts, "Right Ankle")
    LF = get_pt(pts, "Left Foot")
    RF = get_pt(pts, "Right Foot")

    shoulder_mid = midpoint(LS, RS)
    hip_mid = midpoint(LH, RH)

    knee_left = ang.get("knee_left")
    knee_right = ang.get("knee_right")
    knee_mean = ang.get("knee_mean")

    if knee_left is None and LH is not None and LK is not None and LA is not None:
        knee_left = joint_angle(LH, LK, LA)
        ang["knee_left"] = knee_left

    if knee_right is None and RH is not None and RK is not None and RA is not None:
        knee_right = joint_angle(RH, RK, RA)
        ang["knee_right"] = knee_right

    if knee_mean is None and knee_left is not None and knee_right is not None:
        ang["knee_mean"] = float((knee_left + knee_right) / 2)

    trunk = None if (shoulder_mid is None or hip_mid is None) else (shoulder_mid - hip_mid)

    hip_left = None
    hip_right = None

    if trunk is not None and LK is not None and hip_mid is not None:
        thigh_L = LK - hip_mid
        hip_left = angle_between(trunk, thigh_L)
    if trunk is not None and RK is not None and hip_mid is not None:
        thigh_R = RK - hip_mid
        hip_right = angle_between(trunk, thigh_R)

    ang["hip_flexion_left"] = hip_left
    ang["hip_flexion_right"] = hip_right
    ang["hip_flexion_mean"] = None if (hip_left is None or hip_right is None) else float((hip_left + hip_right) / 2)

    ankle_left = joint_angle(LK, LA, LF) if (LK is not None and LA is not None and LF is not None) else None
    ankle_right = joint_angle(RK, RA, RF) if (RK is not None and RA is not None and RF is not None) else None

    ang["ankle_dorsi_left"] = ankle_left
    ang["ankle_dorsi_right"] = ankle_right
    ang["ankle_dorsi_mean"] = None if (ankle_left is None or ankle_right is None) else float((ankle_left + ankle_right) / 2)

    ang["trunk_inclination"] = angle_between(trunk, V_UP) if trunk is not None else None

    tibia_left = angle_between((LK - LA), V_UP) if (LK is not None and LA is not None) else None
    tibia_right = angle_between((RK - RA), V_UP) if (RK is not None and RA is not None) else None

    ang["tibia_inclination_left"] = tibia_left
    ang["tibia_inclination_right"] = tibia_right
    ang["tibia_inclination_mean"] = None if (tibia_left is None or tibia_right is None) else float((tibia_left + tibia_right) / 2)

    fppa_left = angle_between((LH - LK), (LA - LK)) if (LH is not None and LK is not None and LA is not None) else None
    fppa_right = angle_between((RH - RK), (RA - RK)) if (RH is not None and RK is not None and RA is not None) else None

    ang["fppa_left"] = fppa_left
    ang["fppa_right"] = fppa_right
    ang["fppa_mean"] = None if (fppa_left is None or fppa_right is None) else float((fppa_left + fppa_right) / 2)

    pelvis_vec = (RH - LH) if (LH is not None and RH is not None) else None
    ang["pelvic_tilt"] = angle_between(pelvis_vec, H_RIGHT) if pelvis_vec is not None else None

    shoulder_line = (RS - LS) if (LS is not None and RS is not None) else None
    ang["trunk_lateral_tilt"] = angle_between(shoulder_line, H_RIGHT) if shoulder_line is not None else None

    d_knee = dist(LK, RK)
    d_ankle = dist(LA, RA)

    if d_knee is None or d_ankle is None or d_ankle == 0:
        ang["knee_width_ratio"] = None
    else:
        ang["knee_width_ratio"] = float(d_knee / d_ankle)

    head_pitch_proxy = None
    if Nose is not None and LEye is not None and REye is not None and LS is not None and RS is not None:
        face_mid = (Nose + LEye + REye) / 3.0
        shoulder_mid2 = (LS + RS) / 2.0
        d_face_shoulder = np.linalg.norm(face_mid - shoulder_mid2)
        shoulder_width = np.linalg.norm(RS - LS)
        if shoulder_width > 1e-6:
            head_pitch_proxy = float(d_face_shoulder / shoulder_width)
    ang["head_pitch_proxy"] = head_pitch_proxy

    face_shoulder_hip_left = None
    face_shoulder_hip_right = None

    if Nose is not None and LEye is not None and REye is not None:
        face_mid = (Nose + LEye + REye) / 3.0

        if face_mid is not None and LS is not None and LH is not None:
            face_shoulder_hip_left = joint_angle(face_mid, LS, LH)

        if face_mid is not None and RS is not None and RH is not None:
            face_shoulder_hip_right = joint_angle(face_mid, RS, RH)

    ang["face_shoulder_hip_left"] = face_shoulder_hip_left
    ang["face_shoulder_hip_right"] = face_shoulder_hip_right

    return ang


# ---------- 클래스 ----------


class VideoPoseExtractor:
    def __init__(self, video_path: str) -> None:
        self.video_path = video_path
        self.width = 0
        self.height = 0
        self.fps = 0.0

    def extract_landmarks(self) -> Tuple[np.ndarray | None, np.ndarray | None, List[mp.framework.formats.landmark_pb2.NormalizedLandmarkList | None]]:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"❌ Error: Cannot open video {self.video_path}")
            return None, None, []

        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(cap.get(cv2.CAP_PROP_FPS))

        coords_list: List[np.ndarray] = []
        vis_list: List[np.ndarray] = []
        raw_list: List[mp.framework.formats.landmark_pb2.NormalizedLandmarkList | None] = []

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as pose:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                f_coords = np.full((len(KEYPOINTS), 2), np.nan, dtype=np.float32)
                f_vis = np.full((len(KEYPOINTS),), 0.0, dtype=np.float32)

                if results.pose_landmarks:
                    raw_list.append(results.pose_landmarks)
                    for i, key in enumerate(KEYPOINTS):
                        lm = results.pose_landmarks.landmark[MP_MAP[key]]
                        f_coords[i] = [lm.x * self.width, lm.y * self.height]
                        f_vis[i] = lm.visibility
                else:
                    raw_list.append(None)

                coords_list.append(f_coords)
                vis_list.append(f_vis)

        cap.release()
        return np.array(coords_list), np.array(vis_list), raw_list


class VideoVisualizer:
    def save_skeleton_video(self, input_path: str, output_path: str, raw_landmarks: Sequence[mp.framework.formats.landmark_pb2.NormalizedLandmarkList | None]) -> None:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if idx < len(raw_landmarks) and raw_landmarks[idx]:
                mp_drawing.draw_landmarks(
                    frame,
                    raw_landmarks[idx],
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing_styles.get_default_pose_landmarks_style(),
                )
            out.write(frame)
            idx += 1

        cap.release()
        out.release()
        print(f"🎥 [Saved] 스켈레톤 영상 저장 완료: {output_path}")

    def save_rep_clip(
        self,
        input_path: str,
        output_path: str,
        raw_landmarks: Sequence[mp.framework.formats.landmark_pb2.NormalizedLandmarkList | None],
        frame_range: Tuple[int, int],
        loop_count: int = 2,
    ) -> None:
        start, end = frame_range
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        frames_to_copy: List[np.ndarray] = []
        for frame_idx in range(start, end + 1):
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx < len(raw_landmarks) and raw_landmarks[frame_idx]:
                mp_drawing.draw_landmarks(
                    frame,
                    raw_landmarks[frame_idx],
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing_styles.get_default_pose_landmarks_style(),
                )
            frames_to_copy.append(frame)

        for _ in range(max(loop_count, 1)):
            for frame in frames_to_copy:
                out.write(frame)

        cap.release()
        out.release()
        print(f"🎞️ [Saved] Rep 클립 저장 완료: {output_path}")


class PoseAnalyzer:
    def __init__(self, smooth_window: int = 5) -> None:
        self.smooth_window = smooth_window
        self.replacement_logs: List[str] = []
        self.kp_idx = {k: i for i, k in enumerate(KEYPOINTS)}

    def smooth_signal(self, data: np.ndarray) -> np.ndarray:
        if len(data) < self.smooth_window:
            return data
        window = np.ones(self.smooth_window) / self.smooth_window
        return np.convolve(data, window, mode="same")

    def _array_to_pts_dict(self, frame_coords: np.ndarray) -> Dict[str, Dict[str, float]]:
        pts: Dict[str, Dict[str, float]] = {}
        for i, k in enumerate(KEYPOINTS):
            if not np.isnan(frame_coords[i]).any():
                pts[k] = {"x": float(frame_coords[i][0]), "y": float(frame_coords[i][1])}
        return pts

    def calculate_angles_and_filter(
        self, coords: np.ndarray, vis: np.ndarray, view_mode: str = "side", vis_threshold: float = 0.6
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        records: List[Dict[str, float]] = []
        self.replacement_logs = []

        idx_lk = self.kp_idx["Left Knee"]
        idx_rk = self.kp_idx["Right Knee"]

        for t in range(len(coords)):
            pts = self._array_to_pts_dict(coords[t])
            angles = compute_all_angles(pts)

            if view_mode == "side":
                vis_l = vis[t][idx_lk]
                vis_r = vis[t][idx_rk]
                knee_l = angles.get("knee_left")
                knee_r = angles.get("knee_right")
                smart_mean = angles.get("knee_mean")

                if (knee_l is not None and vis_l >= vis_threshold) and (knee_r is None or vis_r < vis_threshold):
                    smart_mean = knee_l
                    self.replacement_logs.append(
                        f"Frame {t}: Right Knee Hidden (Vis {vis_r:.2f}) -> Replaced with Left"
                    )
                elif (knee_r is not None and vis_r >= vis_threshold) and (knee_l is None or vis_l < vis_threshold):
                    smart_mean = knee_r
                    self.replacement_logs.append(
                        f"Frame {t}: Left Knee Hidden (Vis {vis_l:.2f}) -> Replaced with Right"
                    )

                angles["knee_mean"] = smart_mean

            if angles.get("knee_mean") is None:
                angles["knee_mean"] = 180.0

            records.append(angles)

        df_all = pd.DataFrame(records)

        if view_mode == "side":
            target_cols = SIDE_VIEW_COLUMNS
        elif view_mode == "front":
            target_cols = FRONT_VIEW_COLUMNS
        else:
            raise ValueError("view_mode must be 'side' or 'front'")

        return df_all, df_all[target_cols]

    def resample_rep(self, df_rep: pd.DataFrame, target_len: int = 64) -> pd.DataFrame:
        if df_rep.empty:
            return df_rep
        x_old = np.linspace(0, 1, len(df_rep))
        x_new = np.linspace(0, 1, target_len)
        df_new = pd.DataFrame()

        for col in df_rep.columns:
            vals = df_rep[col].values.astype(float)
            mask = np.isnan(vals)
            if np.any(mask) and np.any(~mask):
                vals[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), vals[~mask])
            df_new[col] = np.interp(x_new, x_old, vals)
        return df_new


class RepDetector:
    def __init__(self, min_delta: float = 10.0, min_frames: int = 4) -> None:
        self.min_delta = min_delta
        self.min_frames = min_frames

    def detect(self, seq: np.ndarray) -> List[Dict[str, int | float]]:
        if np.isnan(seq).all():
            return []
        peaks: List[int] = []
        bottoms: List[int] = []

        for t in range(1, len(seq) - 1):
            if seq[t] > seq[t - 1] and seq[t] >= seq[t + 1]:
                peaks.append(t)
            elif seq[t] < seq[t - 1] and seq[t] <= seq[t + 1]:
                bottoms.append(t)

        reps: List[Dict[str, int | float]] = []
        peaks_arr = np.array(peaks)

        for b in bottoms:
            prev = peaks_arr[peaks_arr < b]
            nxt = peaks_arr[peaks_arr > b]
            if len(prev) > 0 and len(nxt) > 0:
                s, e = int(prev[-1]), int(nxt[0])
                if (seq[s] - seq[b]) > self.min_delta and (e - s) > self.min_frames:
                    if not reps or reps[-1]["end"] != e:
                        reps.append({"start": s, "bottom": int(b), "end": e, "min_angle": float(seq[b])})
        return reps


@dataclass
class RepSegment:
    rep_id: int
    view_mode: str
    frame_range: Tuple[int, int]
    data: pd.DataFrame
    clip_path: str


@dataclass
class PosePipelineResult:
    reps: List[RepSegment]
    skeleton_video: str
    occlusion_logs: List[str]
    fps: float


def process_video_with_reps(video_path: str, view_mode: str = "side", output_dir: str = "outputs") -> PosePipelineResult:
    print(f"\n🎬 [Start] 영상 분석 시작 (Mode: {view_mode}): {os.path.basename(video_path)}")

    extractor = VideoPoseExtractor(video_path)
    coords, vis, raw_lm = extractor.extract_landmarks()
    if coords is None or vis is None:
        return PosePipelineResult(reps=[], skeleton_video="", occlusion_logs=[], fps=0.0)

    os.makedirs(output_dir, exist_ok=True)

    vis_saver = VideoVisualizer()
    skeleton_output = str(Path(output_dir) / f"{Path(video_path).stem}_{view_mode}_skeleton.mp4")
    vis_saver.save_skeleton_video(video_path, skeleton_output, raw_lm)

    analyzer = PoseAnalyzer()
    coords_smooth = np.zeros_like(coords)
    for i in range(coords.shape[1]):
        for j in range(2):
            coords_smooth[:, i, j] = analyzer.smooth_signal(coords[:, i, j])

    df_all, df_filtered = analyzer.calculate_angles_and_filter(coords_smooth, vis, view_mode=view_mode)

    if view_mode == "side":
        print(f"\n🔍 [Occlusion Log] 가려짐 대체 내역 ({len(analyzer.replacement_logs)}건):")
        if analyzer.replacement_logs:
            for log in analyzer.replacement_logs[:3]:
                print(f"  - {log}")
            if len(analyzer.replacement_logs) > 3:
                print("  ... (생략)")
        else:
            print("  - 가려짐 대체가 발생하지 않았습니다.")

    knee_seq = analyzer.smooth_signal(df_all["knee_mean"].values)
    detector = RepDetector(min_delta=10.0)
    reps = detector.detect(knee_seq)

    print(f"\n✅ [Result] 총 {len(reps)}개의 Rep이 감지되었습니다.")

    rep_segments: List[RepSegment] = []
    for i, rep in enumerate(reps):
        start, end = rep["start"], rep["end"]
        rep_data = analyzer.resample_rep(df_filtered.iloc[start : end + 1], target_len=64)

        clip_path = str(Path(output_dir) / f"{Path(video_path).stem}_{view_mode}_rep{i+1}.mp4")
        vis_saver.save_rep_clip(video_path, clip_path, raw_lm, (start, end))

        rep_segments.append(
            RepSegment(
                rep_id=i + 1,
                view_mode=view_mode,
                frame_range=(start, end),
                data=rep_data,
                clip_path=clip_path,
            )
        )
        print(f"  - Rep {i+1}: Frame {start}~{end}, Min Depth: {rep['min_angle']:.1f}°")

    return PosePipelineResult(
        reps=rep_segments,
        skeleton_video=skeleton_output,
        occlusion_logs=analyzer.replacement_logs,
        fps=extractor.fps,
    )
