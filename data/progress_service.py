from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from threading import Lock
from typing import Any
from uuid import uuid4


@dataclass
class TaskState:
    task_id: str
    kind: str
    title: str
    status: str
    match_id: str = ""
    issue: str = ""
    current_step: str = ""
    current_item_label: str = ""
    current_item_index: int = 0
    completed_items: int = 0
    total_items: int = 0
    message: str = ""
    level: str = "info"
    error: str = ""
    created_at: str = ""
    updated_at: str = ""


_TASKS: dict[str, TaskState] = {}
_TASKS_LOCK = Lock()
_TASK_RETENTION = timedelta(hours=2)


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _parse_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def _cleanup_locked() -> None:
    now = datetime.now()
    expired_ids = []
    for task_id, task in _TASKS.items():
        if task.status in {"pending", "running"}:
            continue
        if not task.updated_at:
            continue
        if now - _parse_time(task.updated_at) > _TASK_RETENTION:
            expired_ids.append(task_id)

    for task_id in expired_ids:
        _TASKS.pop(task_id, None)


def create_task(kind: str, title: str, *, match_id: str = "", issue: str = "") -> str:
    task_id = uuid4().hex
    now_text = _now_text()
    task = TaskState(
        task_id=task_id,
        kind=kind,
        title=title,
        status="pending",
        match_id=match_id,
        issue=issue,
        created_at=now_text,
        updated_at=now_text,
    )
    with _TASKS_LOCK:
        _cleanup_locked()
        _TASKS[task_id] = task
    return task_id


def update_task(task_id: str, **fields: Any) -> None:
    with _TASKS_LOCK:
        task = _TASKS.get(task_id)
        if task is None:
            return
        for key, value in fields.items():
            if hasattr(task, key):
                setattr(task, key, value)
        task.updated_at = _now_text()


def complete_task(task_id: str, message: str = "", level: str = "success") -> None:
    with _TASKS_LOCK:
        task = _TASKS.get(task_id)
        if task is None:
            return
        if task.total_items and task.completed_items < task.total_items:
            task.completed_items = task.total_items
            if task.current_item_index < task.total_items:
                task.current_item_index = task.total_items
        task.status = "completed"
        task.current_step = task.current_step or "任务完成"
        task.message = message or task.message or "任务已完成"
        task.level = level or "success"
        task.error = ""
        task.updated_at = _now_text()


def fail_task(task_id: str, error: str) -> None:
    with _TASKS_LOCK:
        task = _TASKS.get(task_id)
        if task is None:
            return
        task.status = "failed"
        task.error = error
        task.message = error
        task.level = "error"
        task.updated_at = _now_text()


def get_task(task_id: str) -> dict[str, Any] | None:
    with _TASKS_LOCK:
        task = _TASKS.get(task_id)
        if task is None:
            return None
        payload = asdict(task)

    total_items = int(payload.get("total_items") or 0)
    completed_items = int(payload.get("completed_items") or 0)
    percent = 0
    if total_items > 0:
        base = completed_items
        if payload["status"] == "running" and completed_items < total_items:
            base = max(completed_items, int(payload.get("current_item_index") or 0) - 1)
        percent = min(100, int((base / total_items) * 100))
        if payload["status"] == "completed":
            percent = 100
    payload["progress_percent"] = percent
    return payload


__all__ = [
    "complete_task",
    "create_task",
    "fail_task",
    "get_task",
    "update_task",
]
