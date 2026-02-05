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

# Try importing psutil for resource monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)

class CommandSafetyFilter:
    """Blocks dangerous shell commands that could compromise the system or cause hangs."""
    
    # Blocked patterns (blacklist)
    BLOCKED_PATTERNS = [
        r"rm\s+-rf\s+/",          # Root deletion
        r":\(\)\{ :\|:& \};:",    # Fork bomb
        r"mkfs",                  # Filesystem creation
        r"dd\s+if=/dev/zero",     # Disk wiper
        r"shutdown",              # System shutdown
        r"reboot",                # System reboot
        r"chmod\s+-R\s+777\s+/",  # Dangerous permissions
    ]

    @staticmethod
    def is_safe(command: List[str]) -> tuple[bool, str]:
        cmd_str = " ".join(command).lower()
        
        # Check against blacklist
        import re
        for pattern in CommandSafetyFilter.BLOCKED_PATTERNS:
            if re.search(pattern, cmd_str):
                return False, f"Dangerous command pattern detected: {pattern}"
        
        # Check for potentially infinite loops or high-resource commands
        # (Add more logic here as needed)
        
        return True, ""

class TaskStatus(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    REJECTED = "rejected"

class Task:
    def __init__(self, task_id: str, command: List[str], cwd: Optional[str] = None, log_file: Optional[str] = None):
        self.id = task_id
        self.command = command
        self.cwd = cwd
        self.log_file = log_file
        self.status = TaskStatus.RUNNING
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.exit_code: Optional[int] = None
        self.output_buffer = deque(maxlen=2000)  # Last 2000 lines
        self.process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self.timeout_seconds = 900 # 15 minute global timeout
        self.max_cpu_percent = 80.0
        self.max_memory_mb = 2048.0 # 2GB limit per task

    def start(self):
        """Start the process in a background thread."""
        def run():
            try:
                # Use creationflags to prevent console window on Windows
                creation_flags = 0
                if os.name == 'nt':
                    creation_flags = subprocess.CREATE_NO_WINDOW

                # Open log file if specified
                log_f = open(self.log_file, "w", encoding="utf-8", buffering=1) if self.log_file else None

                try:
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

                    # Start resource monitor thread
                    if HAS_PSUTIL:
                        threading.Thread(target=self._monitor_resources, daemon=True).start()

                    # Read output line by line
                    for line in iter(self.process.stdout.readline, ''):
                        with self._lock:
                            self.output_buffer.append(line)
                        
                        if log_f:
                            log_f.write(line)

                    self.process.wait()
                    with self._lock:
                        if self.status == TaskStatus.RUNNING: # Don't override if monitor set it to FAILED
                            self.exit_code = self.process.returncode
                            self.status = TaskStatus.COMPLETED if self.exit_code == 0 else TaskStatus.FAILED
                        self.end_time = time.time()
                    
                    logger.info(f"Task {self.id} finished with exit code {self.exit_code}")
                    self._notify_completion()
                finally:
                    if log_f:
                        log_f.close()

            except Exception as e:
                logger.error(f"Error running task {self.id}: {e}")
                with self._lock:
                    self.status = TaskStatus.FAILED
                    self.end_time = time.time()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def _monitor_resources(self):
        """Monitor CPU and memory usage of the process."""
        if not self.process: return
        
        try:
            p = psutil.Process(self.process.pid)
            while self.status == TaskStatus.RUNNING:
                # 1. Check Global Timeout
                if time.time() - self.start_time > self.timeout_seconds:
                    logger.warning(f"Task {self.id} timed out after {self.timeout_seconds}s")
                    self._fail_and_kill("Global timeout exceeded (15m)")
                    break

                # 2. Check Resource Usage
                try:
                    with p.oneshot():
                        cpu = p.cpu_percent(interval=None)
                        mem_mb = p.memory_info().rss / (1024 * 1024)
                        
                        if mem_mb > self.max_memory_mb:
                            logger.error(f"Task {self.id} exceeded memory limit: {mem_mb:.1f}MB")
                            self._fail_and_kill(f"Memory limit exceeded ({self.max_memory_mb}MB)")
                            break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                
                time.sleep(5)
        except Exception as e:
            logger.debug(f"Resource monitor error: {e}")

    def _fail_and_kill(self, reason: str):
        with self._lock:
            self.status = TaskStatus.FAILED
            self.exit_code = -1
            self.output_buffer.append(f"\n[RESOURCE GUARD] Task killed: {reason}\n")
        
        if self.process:
            try:
                self.process.kill()
            except (OSError, ProcessLookupError):
                pass  # Process already terminated

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
                
                # Ensure logs directory exists
                cls._instance.logs_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "data", "logs"
                )
                os.makedirs(cls._instance.logs_dir, exist_ok=True)
                
        return cls._instance

    def set_on_complete_callback(self, callback):
        self.on_complete_callback = callback

    def get_system_health(self) -> Dict[str, Any]:
        """Returns PC resource usage to help app decide if it should offload more work."""
        health = {"status": "healthy"}
        if HAS_PSUTIL:
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            health["cpu_percent"] = cpu
            health["mem_percent"] = mem
            if cpu > 90 or mem > 90:
                health["status"] = "strained"
        return health

    def start_task(self, command: List[str], cwd: Optional[str] = None, task_id: Optional[str] = None) -> str:
        """Start a new supervised task with safety checks."""
        # 1. Safety Check
        is_safe, error_msg = CommandSafetyFilter.is_safe(command)
        if not is_safe:
            logger.error(f"Command rejected: {error_msg}")
            tid = task_id or str(uuid.uuid4())
            rejected_task = Task(tid, command, cwd)
            rejected_task.status = TaskStatus.REJECTED
            rejected_task.output_buffer.append(f"ERROR: {error_msg}")
            with self._tasks_lock:
                self.tasks[tid] = rejected_task
            return tid

        # 2. Resource Check
        health = self.get_system_health()
        if health["status"] == "strained":
            logger.warning("PC is strained, delaying task start")
            # For now, we still start it, but in Phase 3/4 we could queue it.

        tid = task_id or str(uuid.uuid4())
        
        # Ensure log directory for this specific task
        task_log_dir = os.path.join(self.logs_dir, tid)
        os.makedirs(task_log_dir, exist_ok=True)
        log_file = os.path.join(task_log_dir, "raw_output.log")
        
        task = Task(tid, command, cwd, log_file=log_file)
        with self._tasks_lock:
            self.tasks[tid] = task
        
        task.start()
        logger.info(f"Started task {tid}: {' '.join(command)}")
        return tid

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

