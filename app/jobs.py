"""Job models and in-memory store (MVP; Redis/DB later)."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    error = "error"


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.queued
    progress: int = 0
    step: str = "В очереди"
    error: str | None = None
    needs_scale: bool = False
    scale_warning: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    drawing_paths: list[Path] = field(default_factory=list)
    photo_paths: list[Path] = field(default_factory=list)
    result_path: Path | None = None
    dxf_path: Path | None = None
    package_path: Path | None = None
    scene_path: Path | None = None
    created_at: float = field(default_factory=time.time)

    def to_public(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "status": self.status.value,
            "progress": self.progress,
            "step": self.step,
            "error": self.error,
            "needs_scale": self.needs_scale,
            "scale_warning": self.scale_warning,
            "has_result": self.result_path is not None and self.result_path.exists(),
            "has_dxf": self.dxf_path is not None and self.dxf_path.exists(),
            "format": "dxf",
        }


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, **kwargs: Any) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, **kwargs)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **kwargs: Any) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            for key, value in kwargs.items():
                setattr(job, key, value)
            return job


jobs = JobStore()
