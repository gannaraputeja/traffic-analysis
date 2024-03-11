from typing import Any

from fastapi.routing import APIRouter
from starlette.responses import Response

from app.schemas.msg import Msg
from app.deps.db import CurrentAsyncSession

router = APIRouter()


@router.get("/start/run", response_model=Msg)
def start_run() -> Any:
    return {"msg": "Hello world!"}
