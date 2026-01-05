"""
Task Service for supervised background processes.
Enables processes to survive disconnects and provide re-attachable output.
"""
import subprocess
import threading
import time
import os
import uuid
import logging
from collections import deque
from enum import Enum
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"

class Task:
    def __init__(self, task_id: str, command: List[str], cwd: Optional[str] = None):
        self.id = task_id
        self.command = command
        self.cwd = cwd
        self.status = TaskStatus.RUNNING
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.exit_code: Optional[int] = None
        self.output_buffer = deque(maxlen=2000)  # Last 2000 lines
        self.process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start the process in a background thread."""
        def run():
            try:
                # Use creationflags to prevent console window on Windows
                creation_flags = 0
                if os.name == 'nt':
                    creation_flags = subprocess.CREATE_NO_WINDOW

                self.process = subprocess.Popen(
                    self.command,
                    cwd=self.cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # Merge stderr into stdout
                    text=True,
                    bufsize=1,  # Line buffered
                    creationflags=creation_flags,
                    env=os.environ.copy()
                )

                # Read output line by line
                for line in iter(self.process.stdout.readline, ''):
                    with self._lock:
                        self.output_buffer.append(line)
                    # logger.debug(f"Task {self.id} output: {line.strip()}")

                self.process.wait()
                with self._lock:
                    self.exit_code = self.process.returncode
                    self.status = TaskStatus.COMPLETED if self.exit_code == 0 else TaskStatus.FAILED
                    self.end_time = time.time()
                
                logger.info(f"Task {self.id} finished with exit code {self.exit_code}")
                
                # Notify completion (could use a callback or event bus)
                self._notify_completion()

            except Exception as e:
                logger.error(f"Error running task {self.id}: {e}")
                with self._lock:
                    self.status = TaskStatus.FAILED
                    self.end_time = time.time()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the process."""
        if self.process and self.status == TaskStatus.RUNNING:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            except Exception as e:
                logger.error(f"Error stopping task {self.id}: {e}")
            
            with self._lock:
                self.status = TaskStatus.STOPPED
                self.end_time = time.time()

    def get_output(self, offset: int = 0) -> List[str]:
        """Get output lines starting from offset."""
        with self._lock:
            lines = list(self.output_buffer)
            if offset < 0:
                return lines # Return all if offset is negative
            return lines[offset:]

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "id": self.id,
                "command": self.command,
                "cwd": self.cwd,
                "status": self.status.value,
                "start_time": self.start_time,
                "end_time": self.end_time,
                "exit_code": self.exit_code,
                "output_count": len(self.output_buffer)
            }

    def _notify_completion(self):
        """Notify Android via companion relay (Phase 4 requirement)."""
        try:
            service = TaskService()
            if service.on_complete_callback:
                service.on_complete_callback(self)
            
            logger.info(f"Task {self.id} complete notification triggered")
        except Exception as e:
            logger.error(f"Failed to notify completion for task {self.id}: {e}")

class TaskService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TaskService, cls).__new__(cls)
                cls._instance.tasks = {}
                cls._instance._tasks_lock = threading.Lock()
                cls._instance.on_complete_callback = None
        return cls._instance

    def set_on_complete_callback(self, callback):
        self.on_complete_callback = callback

    def start_task(self, command: List[str], cwd: Optional[str] = None) -> str:
        """Start a new supervised task."""
        task_id = str(uuid.uuid4())
        task = Task(task_id, command, cwd)
        with self._tasks_lock:
            self.tasks[task_id] = task
        
        task.start()
        logger.info(f"Started task {task_id}: {' '.join(command)}")
        return task_id

    def get_task(self, task_id: str) -> Optional[Task]:
        with self._tasks_lock:
            return self.tasks.get(task_id)

    def stop_task(self, task_id: str) -> bool:
        task = self.get_task(task_id)
        if task:
            task.stop()
            return True
        return False

    def list_tasks(self) -> List[Dict[str, Any]]:
        with self._tasks_lock:
            return [t.to_dict() for t in self.tasks.values()]

    def cleanup_tasks(self, max_age_hours: int = 24):
        """Remove old tasks that are finished."""
        now = time.time()
        with self._tasks_lock:
            to_remove = []
            for tid, task in self.tasks.items():
                if task.status != TaskStatus.RUNNING:
                    if task.end_time and (now - task.end_time) > (max_age_hours * 3600):
                        to_remove.append(tid)
            
            for tid in to_remove:
                del self.tasks[tid]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} finished tasks")

def get_task_service():
    return TaskService()
