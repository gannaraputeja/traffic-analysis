from typing import Any

from fastapi import BackgroundTasks
from fastapi.routing import APIRouter
from starlette.responses import Response

from app.schemas.msg import Msg
from app.yolo.ultralytics import VideoProcessor

router = APIRouter()


@router.get("/start/run", response_model=Msg)
async def start_run(background_tasks: BackgroundTasks) -> Any:
    processor = VideoProcessor(
        source_weights_path='data/traffic_analysis.pt',
        source_video_path='data/traffic_analysis.mov',
        target_video_path='data/traffic_analysis_result.mov',
        confidence_threshold=0.3,
        iou_threshold=0.5,
    )
    background_tasks.add_task(processor.process_video)
    return {"msg": "Started processing video in the background."}
