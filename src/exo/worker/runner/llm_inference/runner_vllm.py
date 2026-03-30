"""vLLM runner — true pipeline-parallel inference across multiple machines.

Each node in the pipeline loads only its assigned layer shard into memory via
Ray + vLLM, enabling models larger than a single machine's RAM to run distributed.

Architecture:
- **Rank 0**: starts a Ray head node, then ``vllm serve`` as the coordinator.
  Forwards TextGeneration tasks to vLLM's OpenAI-compatible HTTP API.
- **Rank 1+**: join the Ray cluster as workers.  vLLM uses them automatically as
  pipeline stages via Ray.  No local inference code needed.
"""

import contextlib
import subprocess
import time
from typing import Literal

import httpx
from loguru import logger

from exo.shared.constants import EXO_DEFAULT_MODELS_DIR
from exo.shared.types.chunks import ErrorChunk, TokenChunk
from exo.shared.types.common import ModelId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.tasks import (
    ConnectToGroup,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
    TextGeneration,
)
from exo.shared.types.worker.instances import BoundInstance, VllmInstance
from exo.shared.types.worker.runners import (
    RunnerConnected,
    RunnerConnecting,
    RunnerFailed,
    RunnerIdle,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerStatus,
    RunnerWarmingUp,
)
from exo.utils.channels import MpReceiver, MpSender

# How long to wait for vllm serve to become ready (seconds)
_VLLM_STARTUP_TIMEOUT = 600
_RAY_STARTUP_TIMEOUT = 60


class VllmRunner:
    """Inference runner backed by vLLM with Ray pipeline parallelism."""

    def __init__(
        self,
        bound_instance: BoundInstance,
        event_sender: MpSender[Event],
        task_receiver: MpReceiver[Task],
        cancel_receiver: MpReceiver[TaskId],
    ):
        self.event_sender = event_sender
        self.task_receiver = task_receiver
        self.cancel_receiver = cancel_receiver
        self.bound_instance = bound_instance

        assert isinstance(bound_instance.instance, VllmInstance)
        self.vllm_instance: VllmInstance = bound_instance.instance

        self.runner_id = bound_instance.bound_runner_id
        self.shard = bound_instance.bound_shard
        self.model_id: ModelId = self.shard.model_card.model_id
        self.device_rank: int = self.shard.device_rank

        self._ray_proc: subprocess.Popen[bytes] | None = None
        self._vllm_proc: subprocess.Popen[bytes] | None = None
        self._cancelled_tasks: set[TaskId] = set()
        self.seen: set[TaskId] = set()

        logger.info(f"VllmRunner created (rank {self.device_rank})")
        self.update_status(RunnerIdle())

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def update_status(self, status: RunnerStatus) -> None:
        self.current_status = status
        self.event_sender.send(
            RunnerStatusUpdated(runner_id=self.runner_id, runner_status=status)
        )

    def send_task_status(self, task_id: TaskId, task_status: TaskStatus) -> None:
        self.event_sender.send(
            TaskStatusUpdated(task_id=task_id, task_status=task_status)
        )

    def acknowledge_task(self, task: Task) -> None:
        self.event_sender.send(TaskAcknowledged(task_id=task.task_id))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def main(self) -> None:
        with self.task_receiver:
            for task in self.task_receiver:
                if task.task_id in self.seen:
                    logger.warning("repeat task - potential error")
                    continue
                self.seen.add(task.task_id)
                self.handle_task(task)
                if isinstance(self.current_status, RunnerShutdown):
                    break

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def handle_task(self, task: Task) -> None:
        self.send_task_status(task.task_id, TaskStatus.Running)

        match task:
            case ConnectToGroup() if isinstance(
                self.current_status, (RunnerIdle, RunnerFailed)
            ):
                self.update_status(RunnerConnecting())
                self.acknowledge_task(task)
                self._connect_to_group()
                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerConnected())

            case LoadModel() if isinstance(
                self.current_status, (RunnerConnected, RunnerIdle)
            ):
                total_layers = self.shard.end_layer - self.shard.start_layer
                self.update_status(RunnerLoading(layers_loaded=0, total_layers=total_layers))
                self.acknowledge_task(task)
                if self.device_rank == 0:
                    self._load_model()
                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerLoaded())

            case StartWarmup() if isinstance(self.current_status, RunnerLoaded):
                self.update_status(RunnerWarmingUp())
                self.acknowledge_task(task)
                if self.device_rank == 0:
                    self._warmup()
                self.send_task_status(task.task_id, TaskStatus.Complete)
                self.update_status(RunnerReady())

            case TextGeneration() if isinstance(
                self.current_status, (RunnerReady, RunnerRunning)
            ):
                if self.device_rank == 0:
                    self.update_status(RunnerRunning())
                    self.acknowledge_task(task)
                    self._run_generation(task)
                    self.update_status(RunnerReady())
                # rank 1+ are passive — vLLM manages them via Ray

            case Shutdown():
                self._shutdown(task)

            case _:
                raise ValueError(
                    f"Received {task.__class__.__name__} outside of state machine "
                    f"in {self.current_status=}"
                )

    # ------------------------------------------------------------------
    # Ray cluster setup
    # ------------------------------------------------------------------

    def _connect_to_group(self) -> None:
        if self.device_rank == 0:
            logger.info("rank 0: starting Ray head node")
            self._ray_proc = subprocess.Popen(
                [
                    "ray", "start", "--head",
                    f"--port={self.vllm_instance.ray_port}",
                    "--disable-usage-stats",
                    "--block",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Wait for Ray head to be accepting connections
            self._wait_for_ray_head()
            logger.info("rank 0: Ray head is up")
        else:
            head_addr = self.vllm_instance.ray_head_address
            logger.info(f"rank {self.device_rank}: joining Ray cluster at {head_addr}")
            self._ray_proc = subprocess.Popen(
                [
                    "ray", "start",
                    f"--address={head_addr}",
                    "--disable-usage-stats",
                    "--block",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Give the worker a moment to register with the head
            time.sleep(3.0)
            logger.info(f"rank {self.device_rank}: joined Ray cluster")

    def _wait_for_ray_head(self) -> None:
        """Poll until the Ray head node is accepting connections."""
        import socket
        deadline = time.time() + _RAY_STARTUP_TIMEOUT
        host, port_str = self.vllm_instance.ray_head_address.rsplit(":", 1)
        port = int(port_str)
        while time.time() < deadline:
            try:
                with socket.create_connection((host, port), timeout=1.0):
                    return
            except OSError:
                time.sleep(0.5)
        raise RuntimeError(
            f"Ray head did not start within {_RAY_STARTUP_TIMEOUT}s "
            f"at {self.vllm_instance.ray_head_address}"
        )

    # ------------------------------------------------------------------
    # Model loading (rank 0 only)
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Resolve the model path and launch ``vllm serve``."""
        model_path = self._resolve_model_path()
        pipeline_parallel_size = self.vllm_instance.pipeline_parallel_size
        port = self.vllm_instance.vllm_serve_port

        logger.info(
            f"rank 0: starting vllm serve "
            f"(model={model_path}, pipeline_parallel_size={pipeline_parallel_size}, port={port})"
        )

        cmd = [
            "vllm", "serve", str(model_path),
            "--port", str(port),
            "--pipeline-parallel-size", str(pipeline_parallel_size),
            "--tensor-parallel-size", "1",
            "--trust-remote-code",
            "--disable-log-requests",
        ]

        self._vllm_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait for vLLM to be ready
        self._wait_for_vllm()
        logger.info("rank 0: vllm serve is ready")

    def _resolve_model_path(self) -> str:
        """Return the model path or ID to pass to ``vllm serve``.

        Preference order:
        1. ``hf_base_model_id`` from the model card (downloads to HF cache automatically)
        2. Local GGUF file (if already downloaded for llama-cpp backend)
        3. ``model_id`` as a fallback (vLLM will attempt to pull from HuggingFace)
        """
        card = self.shard.model_card

        # 1. HF base model ID — vLLM downloads it to the HF cache if not present
        if card.hf_base_model_id is not None:
            normalized = str(card.hf_base_model_id).replace("/", "--")
            local_dir = EXO_DEFAULT_MODELS_DIR / normalized
            if local_dir.exists():
                logger.info(f"Using locally cached HF model at {local_dir}")
                return str(local_dir)
            logger.info(
                f"Model {card.hf_base_model_id} not cached locally; "
                "vllm will download from HuggingFace"
            )
            return str(card.hf_base_model_id)

        # 2. GGUF file (single-file, vLLM supports GGUF for single-node)
        if card.gguf_repo_id is not None and card.gguf_filename is not None:
            dir_name = str(card.gguf_repo_id).replace("/", "--")
            gguf_path = EXO_DEFAULT_MODELS_DIR / dir_name / card.gguf_filename
            if gguf_path.exists():
                logger.info(f"Using local GGUF file at {gguf_path}")
                return str(gguf_path)

        # 3. Fallback: use model_id and let vLLM figure it out
        logger.warning(
            f"No local model found for {card.model_id}; "
            "vllm will try to download from HuggingFace using model_id"
        )
        return str(card.model_id)

    def _wait_for_vllm(self) -> None:
        """Poll the vLLM health endpoint until it responds."""
        url = f"http://127.0.0.1:{self.vllm_instance.vllm_serve_port}/health"
        deadline = time.time() + _VLLM_STARTUP_TIMEOUT
        while time.time() < deadline:
            if self._vllm_proc is not None and self._vllm_proc.poll() is not None:
                raise RuntimeError("vllm serve process exited unexpectedly during startup")
            try:
                resp = httpx.get(url, timeout=2.0)
                if resp.status_code == 200:
                    return
            except (httpx.ConnectError, httpx.TimeoutException):
                pass
            time.sleep(2.0)
        raise RuntimeError(
            f"vllm serve did not become ready within {_VLLM_STARTUP_TIMEOUT}s"
        )

    # ------------------------------------------------------------------
    # Warmup (rank 0 only)
    # ------------------------------------------------------------------

    def _warmup(self) -> None:
        port = self.vllm_instance.vllm_serve_port
        try:
            httpx.post(
                f"http://127.0.0.1:{port}/v1/completions",
                json={"model": self._resolve_model_path(), "prompt": "Hello", "max_tokens": 1},
                timeout=30.0,
            )
            logger.info("vLLM warmup complete")
        except Exception as exc:
            logger.warning(f"vLLM warmup failed (non-fatal): {exc}")

    # ------------------------------------------------------------------
    # Generation (rank 0 only)
    # ------------------------------------------------------------------

    def _run_generation(self, task: TextGeneration) -> None:
        port = self.vllm_instance.vllm_serve_port
        params = task.task_params
        command_id = task.command_id

        messages: list[dict[str, str]] = []
        if params.instructions:
            messages.append({"role": "system", "content": params.instructions})
        for msg in params.input:
            messages.append({"role": msg.role, "content": msg.content})

        payload: dict[str, object] = {
            "model": self._resolve_model_path(),
            "messages": messages,
            "stream": True,
            "max_tokens": params.max_output_tokens or 2048,
        }
        if params.temperature is not None:
            payload["temperature"] = params.temperature
        if params.top_p is not None:
            payload["top_p"] = params.top_p
        # Merge caller-supplied stops with common end-of-turn tokens so vLLM
        # always halts at turn boundaries regardless of which model is loaded.
        eot_stop_strings = [
            "<|eot_id|>",       # Llama 3.x
            "<|end_of_text|>",  # Llama 3.x
            "<|im_end|>",       # Qwen / ChatML
            "<|endoftext|>",    # GPT-2 / Falcon
        ]
        stop_strings = list(dict.fromkeys(list(params.stop or []) + eot_stop_strings))
        payload["stop"] = stop_strings

        try:
            with httpx.Client(timeout=None) as client, client.stream(
                "POST",
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json=payload,
            ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if task.task_id in self._cancelled_tasks:
                            break
                        if not line.startswith("data: "):
                            continue
                        data = line[len("data: "):]
                        if data.strip() == "[DONE]":
                            break
                        import json as _json
                        try:
                            chunk_data = _json.loads(data)  # type: ignore[var-annotated]
                        except Exception:
                            continue
                        choices = chunk_data.get("choices") or []  # type: ignore[union-attr]
                        if not choices:
                            continue
                        choice = choices[0]  # type: ignore[index]
                        delta = choice.get("delta") or {}  # type: ignore[union-attr]
                        text: str = str(delta.get("content") or "")  # type: ignore[union-attr]
                        _fr_raw: object = choice.get("finish_reason")  # type: ignore[union-attr]
                        _fr_str: str | None = _fr_raw if isinstance(_fr_raw, str) else None
                        _fr_map: dict[str, Literal["stop", "length", "content_filter"]] = {
                            "stop": "stop", "length": "length", "content_filter": "content_filter"
                        }
                        finish_reason: Literal["stop", "length", "content_filter"] | None = (
                            _fr_map.get(_fr_str) if _fr_str is not None else None
                        )
                        if text or finish_reason is not None:
                            self.event_sender.send(
                                ChunkGenerated(
                                    command_id=command_id,
                                    chunk=TokenChunk(
                                        model=self.model_id,
                                        text=text,
                                        token_id=0,
                                        usage=None,
                                        finish_reason=finish_reason,
                                        stats=None,
                                        logprob=None,
                                        top_logprobs=None,
                                        is_thinking=False,
                                    ),
                                )
                            )
        except Exception as exc:
            logger.opt(exception=exc).error("vLLM generation error")
            self.event_sender.send(
                ChunkGenerated(
                    command_id=command_id,
                    chunk=ErrorChunk(
                        error_message=str(exc),
                        model=self.model_id,
                    ),
                )
            )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self, task: Task) -> None:
        logger.info("vLLM runner shutting down")
        self.update_status(RunnerShuttingDown())
        self.acknowledge_task(task)

        for _, proc in [("vllm", self._vllm_proc), ("ray", self._ray_proc)]:
            if proc is not None:
                try:
                    proc.terminate()
                    proc.wait(timeout=10)
                except Exception:
                    with contextlib.suppress(Exception):
                        proc.kill()

        # Also stop Ray via CLI to ensure clean teardown
        with contextlib.suppress(Exception):
            subprocess.run(["ray", "stop", "--force"], timeout=10, check=False)

        self.send_task_status(task.task_id, TaskStatus.Complete)
        self.update_status(RunnerShutdown())
