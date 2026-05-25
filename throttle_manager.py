import asyncio
import logging
from typing import Optional, Dict

logger = logging.getLogger("sampling_proxy.throttle")

class ModelThrottle:
    """Holds throttle state for a single model."""

    def __init__(self, start_pause: Optional[float], end_pause: Optional[float]):
        self.start_pause = start_pause
        self.end_pause = end_pause
        self.start_semaphore = asyncio.Semaphore(1) if start_pause else None
        self.end_semaphore = asyncio.Semaphore(1) if end_pause else None

class ThrottleManager:
    """Manages throttle timers for backend requests."""

    def __init__(self, config: dict, enable_debug_logs: bool = False, request_id: int = 0):
        enabled = config.get("enabled", False)
        if enabled:
            self._validate_config(config)

        self.enabled = enabled
        self.global_config = config.get("global", {})
        self.per_model_config = config.get("per_model", {})
        self.lock = asyncio.Lock()
        self.model_state: Dict[str, ModelThrottle] = {}
        self._background_tasks: set = set()
        self.enable_debug_logs = enable_debug_logs
        self.request_id = request_id

    def _validate_config(self, config: dict):
        """Validate throttle configuration."""
        global_config = config.get("global", {})
        self._validate_pause_config(global_config, "global")

        per_model = config.get("per_model", {})
        for model_name, model_config in per_model.items():
            if not isinstance(model_config, dict):
                raise ValueError(f"per_model.{model_name} must be a dict, got {type(model_config).__name__}")
            self._validate_pause_config(model_config, f"per_model.{model_name}")

    def _validate_pause_config(self, pause_config: dict, location: str):
        """Validate a single pause config (global or per-model)."""
        for key in ["start_pause_seconds", "end_pause_seconds"]:
            value = pause_config.get(key)
            if value is not None:
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ValueError(f"{location}.{key} must be a number or null, got {type(value).__name__}")
                if value < 0:
                    raise ValueError(f"{location}.{key} must be >= 0, got {value}")

    def _log_debug(self, message: str, request_id: int = None):
        """Log debug message if enabled."""
        if self.enable_debug_logs:
            rid = request_id if request_id is not None else self.request_id
            logger.debug(f"[R:{rid}] Throttle: {message}")

    def _log_info(self, message: str, request_id: int = None):
        """Log info message."""
        rid = request_id if request_id is not None else self.request_id
        logger.info(f"[R:{rid}] Throttle: {message}")

    # Sentinel indicating "not specified in config" (distinct from None which means "disabled")
    _UNSET = object()

    async def _get_or_create_throttle(self, model: str) -> ModelThrottle:
        """Get or create ModelThrottle for a model.

        Per-model config semantics:
        - Key present with a number: use that value
        - Key present with None: explicitly disable this timer
        - Key absent: fall back to global config
        """
        model_lower = model.lower()

        async with self.lock:
            if model_lower not in self.model_state:
                # Get config for this model
                per_model_config = self.per_model_config.get(model, self.per_model_config.get(model_lower, {}))

                start_pause = per_model_config.get("start_pause_seconds", self._UNSET)
                if start_pause is self._UNSET:
                    start_pause = self.global_config.get("start_pause_seconds")

                end_pause = per_model_config.get("end_pause_seconds", self._UNSET)
                if end_pause is self._UNSET:
                    end_pause = self.global_config.get("end_pause_seconds")

                self.model_state[model_lower] = ModelThrottle(start_pause, end_pause)
                self._log_debug(f"Created throttle state for new model: {model}")

            return self.model_state[model_lower]

    async def wait_before_send(self, model: str, request_id: int = None) -> None:
        """Wait for start-timer before sending request upstream."""
        if not self.enabled:
            return

        throttle = await self._get_or_create_throttle(model)
        if throttle.start_semaphore is None:
            self._log_debug(f"start-timer disabled for {model}", request_id)
            return

        await throttle.start_semaphore.acquire()

        # Schedule auto-release after configured delay
        if throttle.start_pause:
            sem = throttle.start_semaphore
            pause = throttle.start_pause
            rid = request_id
            model_ref = model

            async def release_after_delay():
                try:
                    await asyncio.sleep(pause)
                finally:
                    sem.release()
                    self._log_debug(f"{model_ref} start-timer released after {pause}s", rid)

            task = asyncio.create_task(release_after_delay())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    async def wait_after_send(self, model: str, request_id: int = None) -> None:
        """Wait for end-timer after response completes."""
        if not self.enabled:
            return

        throttle = await self._get_or_create_throttle(model)
        if throttle.end_semaphore is None:
            self._log_debug(f"end-timer disabled for {model}", request_id)
            return

        await throttle.end_semaphore.acquire()

        # Schedule auto-release after configured delay
        if throttle.end_pause:
            sem = throttle.end_semaphore
            pause = throttle.end_pause
            rid = request_id
            model_ref = model

            async def release_after_delay():
                try:
                    await asyncio.sleep(pause)
                finally:
                    sem.release()
                    self._log_debug(f"{model_ref} end-timer released after {pause}s", rid)

            task = asyncio.create_task(release_after_delay())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)