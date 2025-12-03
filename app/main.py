from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .models import RepEvaluation, SquatAnalyzer
from .pose_pipeline import PosePipelineResult, process_video_with_reps


def _rep_to_serializable(rep_eval: RepEvaluation, clip_path: str, frame_range: tuple[int, int]) -> Dict[str, Any]:
    return {
        "rep_id": rep_eval.rep_id,
        "view_mode": rep_eval.view_mode,
        "frame_range": frame_range,
        "clip_path": clip_path,
        "scores": [score.__dict__ for score in rep_eval.scores],
        "probabilities": rep_eval.probabilities,
    }


def analyze_video(video_path: str, view_mode: str, output_dir: str, model_v1: str, model_v3: str) -> Dict[str, Any]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pipeline_result: PosePipelineResult = process_video_with_reps(video_path, view_mode=view_mode, output_dir=output_dir)

    analyzer = SquatAnalyzer(model_v1_path=model_v1, model_v3_path=model_v3)

    rep_summaries: List[Dict[str, Any]] = []
    for rep in pipeline_result.reps:
        records = rep.data.to_dict(orient="records")
        rep_eval: RepEvaluation = analyzer.evaluate_rep(rep.rep_id, records, view_mode=view_mode)
        rep_summaries.append(_rep_to_serializable(rep_eval, rep.clip_path, rep.frame_range))

        print(f"\n📊 Rep {rep.rep_id} 결과 ({view_mode}):")
        for score in rep_eval.scores:
            tip = f"  💡 Tip: {score.tip}" if score.tip else ""
            print(f" - {score.label}: {score.score:.1f}점 {score.grade}{tip}")

    summary = {
        "video": video_path,
        "view_mode": view_mode,
        "skeleton_video": pipeline_result.skeleton_video,
        "reps": rep_summaries,
        "occlusion_logs": pipeline_result.occlusion_logs,
    }

    summary_path = Path(output_dir) / f"{Path(video_path).stem}_{view_mode}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n💾 분석 요약 저장: {summary_path}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="스쿼트 영상 분석 및 UI용 결과 생성")
    parser.add_argument("--video", required=True, help="분석할 영상 파일 경로")
    parser.add_argument("--view", choices=["side", "front"], default="side", help="카메라 구도 (측면/정면)")
    parser.add_argument("--output-dir", default="outputs", help="결과물 저장 디렉터리")
    parser.add_argument("--model-v1", default=str(Path(__file__).resolve().parents[1] / "squat_view1_final_model.pth"))
    parser.add_argument("--model-v3", default=str(Path(__file__).resolve().parents[1] / "squat_view3_final_model.pth"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_video(args.video, args.view, args.output_dir, args.model_v1, args.model_v3)


if __name__ == "__main__":
    main()
