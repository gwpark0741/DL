# 스쿼트 영상 분석 파이프라인

사용자가 업로드한 스쿼트 영상을 구도(측면/정면)에 맞춰 모델로 평가하고, 각 rep마다 스켈레톤이 입혀진 클립과 점수/피드백을 생성하는 파이프라인입니다. 모델은 64프레임 길이의 각도 시퀀스를 입력으로 사용하며, 한 번의 rep에서만 만든 64프레임을 사용하도록 구성되어 있습니다.

## 주요 구성 요소
- `app/pose_pipeline.py`: MediaPipe로 포즈를 추출하고, 각도 계산/스무딩/가려짐 보정, rep 감지(무릎 각도 기반), 64프레임 리샘플링, 스켈레톤 영상/rep 클립 생성.
- `app/models.py`: LSTM 기반 측면(View1)/정면(View3) 모델 로더와 rep별 점수/피드백 생성기.
- `app/main.py`: CLI 엔트리포인트. 영상 경로와 구도를 받아 전체 파이프라인을 실행하고 JSON 요약을 저장.

## 준비 사항
1. Python 3.10+ 환경.
2. 필수 패키지: `torch`, `opencv-python`, `mediapipe`, `pandas`, `numpy` (로컬 환경에서 미리 설치 필요).
3. 모델 가중치 파일: `squat_view1_final_model.pth`, `squat_view3_final_model.pth` (저장소 루트에 위치).

## 실행 방법
```bash
python -m app.main --video /path/to/video.mp4 --view side --output-dir outputs
# 또는 정면 구도
python -m app.main --video /path/to/video.mp4 --view front --output-dir outputs
```

실행 후 `outputs/`에 다음 결과물이 생성됩니다.
- 전체 스켈레톤 영상: `<원본파일명>_<view>_skeleton.mp4`
- rep별 스켈레톤 클립: `<원본파일명>_<view>_rep<번호>.mp4` (루프 재생용으로 중복 프레임을 포함)
- rep별 점수/피드백을 포함한 JSON 요약: `<원본파일명>_<view>_summary.json`

## UI 연동 가이드
- **업로드 및 구도 선택**: 프런트엔드에서 파일 업로드와 `side/front` 라디오 버튼을 제공하고, 선택값을 CLI 호출 인자로 전달합니다.
- **멀티 rep 처리**: 파이프라인이 rep를 자동 감지하고 각각 64프레임으로 리샘플링하므로, 여러 rep을 한 번에 업로드해도 각 rep별 점수/클립을 개별 카드로 표시할 수 있습니다.
- **영상 반복 재생**: 생성된 `rep*.mp4`는 루프 재생을 고려해 동일 프레임을 두 번 이상 기록했습니다. 웹에서는 `<video loop>` 또는 GIF 변환 후 `<img>`로 노출할 수 있습니다.
- **피드백 표시**: JSON 요약의 `scores` 필드를 UI에 바로 바인딩하면 점수/등급/팁을 보여줄 수 있습니다.
- **에러/의존성 처리**: Mediapipe·OpenCV는 로컬에 설치되어 있어야 하며, GPU가 없다면 CPU로 자동 동작합니다.
