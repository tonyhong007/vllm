# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import queue
import signal
import threading
import time
from collections import deque
from collections.abc import Callable, Generator
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from inspect import isclass, signature
from logging import DEBUG
from typing import Any, TypeVar, cast

import msgspec
import torch
import zmq

from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.envs import enable_envs_cache
from vllm.logger import init_logger
from vllm.logging_utils.dump_input import dump_engine_exception
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import engine_receiver_cache_from_config
from vllm.tasks import POOLING_TASKS, SupportedTask
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
from vllm.utils.gc_utils import (
    freeze_gc_heap,
    maybe_attach_gc_debug_callback,
)
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.utils.network_utils import make_zmq_socket
from vllm.utils.system_utils import decorate_logs, set_process_title
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import (
    EngineCoreOutputs,
    EngineCoreRequest,
    EngineCoreRequestType,
    ReconfigureDistributedRequest,
    ReconfigureRankType,
    UtilityOutput,
    UtilityResult,
)
from vllm.v1.engine.utils import (
    EngineHandshakeMetadata,
    EngineZmqAddresses,
    get_device_indices,
)
from vllm.v1.executor import Executor
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.structured_output import StructuredOutputManager
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)

POLLING_TIMEOUT_S = 2.5
HANDSHAKE_TIMEOUT_MINS = 5

_R = TypeVar("_R")  # Return type for collective_rpc


class EngineCore:
    """Inner loop of vLLM's Engine."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        executor_fail_callback: Callable | None = None,
    ):
        # plugins need to be loaded at the engine/scheduler level too
        from vllm.plugins import load_general_plugins

        load_general_plugins()

        self.vllm_config = vllm_config
        if vllm_config.parallel_config.data_parallel_rank == 0:
            logger.info(
                "Initializing a V1 LLM engine (v%s) with config: %s",
                VLLM_VERSION,
                vllm_config,
            )

        self.log_stats = log_stats

        # Setup Model.
        self.model_executor = executor_class(vllm_config)
        if executor_fail_callback is not None:
            self.model_executor.register_failure_callback(executor_fail_callback)

        self.available_gpu_memory_for_kv_cache = -1

        # Setup KV Caches and update CacheConfig after profiling.
        num_gpu_blocks, num_cpu_blocks, kv_cache_config = self._initialize_kv_caches(
            vllm_config
        )

        vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
        vllm_config.cache_config.num_cpu_blocks = num_cpu_blocks
        self.collective_rpc("initialize_cache", args=(num_gpu_blocks, num_cpu_blocks))

        # Initialize SAGE parallel prefill.
        self.sage_parallel_prefill = (
            os.environ.get("ENABLE_SAGE", "False").lower() == "true"
            and os.environ.get("SAGE_PARALLEL_PREFILL", "").lower()
            in ("1", "true")
        )
        if self.sage_parallel_prefill:
            if os.environ.get("SAGE_INSTANCE_RANK") is None:
                raise RuntimeError(
                    "SAGE_PARALLEL_PREFILL requires SAGE_INSTANCE_RANK"
                )
            if os.environ.get("SAGE_HOME_GPU") is None:
                raise RuntimeError(
                    "SAGE_PARALLEL_PREFILL requires SAGE_HOME_GPU"
                )
        self._sage_instance_rank = int(
            os.environ.get("SAGE_INSTANCE_RANK", "0")
        )
        self._sage_home_rank = int(
            os.environ.get("SAGE_HOME_GPU", "0")
        ) if self.sage_parallel_prefill else 0
        # Per-layer transfer: sends KV layer-by-layer via SageKVTransferEngine
        # callback during forward, enabling pipelined early-blend on the home.
        # Default: bulk transfer after forward.
        self._sage_per_layer_transfer = (
            self.sage_parallel_prefill
            and os.environ.get("SAGE_PER_LAYER_TRANSFER", "").lower()
            in ("1", "true")
        )
        self._sage_kv_transfer = None
        self._sage_attention_layer_names: list[str] = []
        if self.sage_parallel_prefill:
            kv_config = vllm_config.kv_transfer_config
            if kv_config is None:
                raise RuntimeError(
                    "SAGE_PARALLEL_PREFILL requires kv_transfer_config"
                )
            is_home = self._sage_instance_rank == self._sage_home_rank
            if self._sage_per_layer_transfer:
                # Per-layer mode: both home and worker get SageKVTransferEngine.
                # Worker sends per-layer via save_kv_layer callback.
                if is_home:
                    from vllm.v1.engine.sage_kv_transfer import (
                        SageKVTransferEngine,
                    )
                    dp_size = vllm_config.parallel_config.data_parallel_size
                    dp_rank = self._sage_instance_rank
                    local_gpu = (
                        vllm_config.parallel_config.data_parallel_rank_local
                        or 0
                    )
                    self._sage_kv_transfer = SageKVTransferEngine(
                        dp_rank=dp_rank,
                        dp_size=dp_size,
                        local_gpu_id=local_gpu,
                        kv_transfer_config=kv_config,
                    )
                    kv_cache_refs = self.collective_rpc("get_kv_caches")
                    if kv_cache_refs and kv_cache_refs[0]:
                        self._sage_kv_transfer.set_kv_caches(kv_cache_refs[0])
                    _input_q = getattr(self, "input_queue", None)
                    if _input_q is not None:
                        self._sage_kv_transfer._nccl_engine._on_recv_callback = (
                            lambda tid: _input_q.put_nowait(
                                ("_SAGE_REMOTE_KV",)
                            ) if "#header" in tid else None
                        )
                    names = self.collective_rpc("get_attention_layer_names")
                    if names and names[0]:
                        self._sage_attention_layer_names = names[0]
                        self._sage_kv_transfer.set_layer_names(
                            self._sage_attention_layer_names
                        )
                else:
                    # Worker: create SageKVTransferEngine for per-layer
                    # NCCL send (replaces P2pNcclConnector approach).
                    from vllm.v1.engine.sage_kv_transfer import (
                        SageKVTransferEngine,
                    )
                    dp_size = vllm_config.parallel_config.data_parallel_size
                    dp_rank = self._sage_instance_rank
                    local_gpu = (
                        vllm_config.parallel_config.data_parallel_rank_local
                        or 0
                    )
                    self._sage_kv_transfer = SageKVTransferEngine(
                        dp_rank=dp_rank,
                        dp_size=dp_size,
                        local_gpu_id=local_gpu,
                        kv_transfer_config=kv_config,
                    )
                    kv_cache_refs = self.collective_rpc("get_kv_caches")
                    if kv_cache_refs and kv_cache_refs[0]:
                        self._sage_kv_transfer.set_kv_caches(kv_cache_refs[0])
                    names = self.collective_rpc("get_attention_layer_names")
                    if names and names[0]:
                        self._sage_attention_layer_names = names[0]
                        self._sage_kv_transfer.set_layer_names(names[0])
                        # Build name→index lookup for the callback.
                        self._sage_layer_name_to_idx = {
                            n: i for i, n in enumerate(names[0])
                        }
                    # Pending per-layer sends for the current step.
                    # Populated before each forward, consumed by the
                    # save_kv_layer callback during forward.
                    self._sage_kv_transfer._pending_sends = []
                    # Register per-layer send callback on the connector.
                    self.collective_rpc(
                        "sage_register_per_layer_callback",
                        args=(self._sage_kv_transfer,
                              self._sage_layer_name_to_idx),
                    )
            else:
                # Bulk mode (default): SageKVTransferEngine on all instances.
                from vllm.v1.engine.sage_kv_transfer import SageKVTransferEngine
                dp_size = vllm_config.parallel_config.data_parallel_size
                dp_rank = self._sage_instance_rank
                local_gpu = (
                    vllm_config.parallel_config.data_parallel_rank_local or 0
                )
                self._sage_kv_transfer = SageKVTransferEngine(
                    dp_rank=dp_rank,
                    dp_size=dp_size,
                    local_gpu_id=local_gpu,
                    kv_transfer_config=kv_config,
                )
                kv_cache_refs = self.collective_rpc("get_kv_caches")
                if kv_cache_refs and kv_cache_refs[0]:
                    self._sage_kv_transfer.set_kv_caches(kv_cache_refs[0])
                _input_q = getattr(self, "input_queue", None)
                if _input_q is not None:
                    self._sage_kv_transfer._nccl_engine._on_recv_callback = (
                        lambda tid: _input_q.put_nowait(
                            ("_SAGE_REMOTE_KV",)
                        ) if "#bulk_kv" in tid else None
                    )

            # Pre-warm the LMCache blender on the home GPU. Saves ~50 ms
            # on first request TTFT by paying the ~200 ms cuBLAS / kernel
            # JIT cold-start cost during engine init instead of on the
            # critical path. Trade-off: ~200 ms added to engine startup,
            # amortized across all requests in the engine's lifetime.
            if is_home:
                try:
                    self.collective_rpc("sage_warmup_blender")
                except Exception as e:
                    logger.warning(
                        "[SAGE_WARMUP] failed (non-fatal): %s", e,
                    )

        self.structured_output_manager = StructuredOutputManager(vllm_config)

        # Setup scheduler.
        Scheduler = vllm_config.scheduler_config.get_scheduler_cls()

        if len(kv_cache_config.kv_cache_groups) == 0:  # noqa: SIM102
            # Encoder models without KV cache don't support
            # chunked prefill. But do SSM models?
            if vllm_config.scheduler_config.enable_chunked_prefill:
                logger.warning("Disabling chunked prefill for model without KVCache")
                vllm_config.scheduler_config.enable_chunked_prefill = False

        scheduler_block_size = (
            vllm_config.cache_config.block_size
            * vllm_config.parallel_config.decode_context_parallel_size
            * vllm_config.parallel_config.prefill_context_parallel_size
        )

        self.scheduler: SchedulerInterface = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=self.structured_output_manager,
            include_finished_set=vllm_config.parallel_config.data_parallel_size > 1,
            log_stats=self.log_stats,
            block_size=scheduler_block_size,
        )
        self.use_spec_decode = vllm_config.speculative_config is not None
        if self.scheduler.connector is not None:  # type: ignore
            self.model_executor.init_kv_output_aggregator(self.scheduler.connector)  # type: ignore

        self.mm_registry = mm_registry = MULTIMODAL_REGISTRY
        self.mm_receiver_cache = engine_receiver_cache_from_config(
            vllm_config, mm_registry
        )

        # If a KV connector is initialized for scheduler, we want to collect
        # handshake metadata from all workers so the connector in the scheduler
        # will have the full context
        kv_connector = self.scheduler.get_kv_connector()
        if kv_connector is not None:
            # Collect and store KV connector xfer metadata from workers
            # (after KV cache registration)
            xfer_handshake_metadata = (
                self.model_executor.get_kv_connector_handshake_metadata()
            )

            if xfer_handshake_metadata:
                # xfer_handshake_metadata is list of dicts from workers
                # Each dict already has structure {tp_rank: metadata}
                # Merge all worker dicts into a single dict
                content: dict[int, Any] = {}
                for worker_dict in xfer_handshake_metadata:
                    if worker_dict is not None:
                        content.update(worker_dict)
                kv_connector.set_xfer_handshake_metadata(content)

        # Setup batch queue for pipeline parallelism.
        # Batch queue for scheduled batches. This enables us to asynchronously
        # schedule and execute batches, and is required by pipeline parallelism
        # to eliminate pipeline bubbles.
        self.batch_queue_size = self.model_executor.max_concurrent_batches
        self.batch_queue: (
            deque[tuple[Future[ModelRunnerOutput], SchedulerOutput]] | None
        ) = None
        if self.batch_queue_size > 1:
            logger.info("Batch queue is enabled with size %d", self.batch_queue_size)
            self.batch_queue = deque(maxlen=self.batch_queue_size)

        self.is_ec_producer = (
            vllm_config.ec_transfer_config is not None
            and vllm_config.ec_transfer_config.is_ec_producer
        )
        self.is_pooling_model = vllm_config.model_config.runner_type == "pooling"

        self.request_block_hasher: Callable[[Request], list[BlockHash]] | None = None
        if vllm_config.cache_config.enable_prefix_caching or kv_connector is not None:
            caching_hash_fn = get_hash_fn_by_name(
                vllm_config.cache_config.prefix_caching_hash_algo
            )
            init_none_hash(caching_hash_fn)

            self.request_block_hasher = get_request_block_hasher(
                scheduler_block_size, caching_hash_fn
            )

        self.step_fn = (
            self.step if self.batch_queue is None else self.step_with_batch_queue
        )
        self.async_scheduling = vllm_config.scheduler_config.async_scheduling

        self.aborts_queue = queue.Queue[list[str]]()

        # Concurrent prefill configuration
        self.enable_sage = os.environ.get(
            "ENABLE_SAGE", "False"
        ).lower() == "true"

        from collections import deque as _sage_deque
        self._sage_pending_parents: "deque[tuple[str, object]]" = (
            _sage_deque()
        )
        self._sage_pending_chunks: "deque[object]" = _sage_deque()
        self._sage_stuck_steps = 0
        self._sage_last_queue_size = -1

        # Remote chunks whose HEADER has been received (home GPU only).
        # parent_id -> {chunk_req_id: (position, dest_block_ids, num_tokens)}
        self._sage_remote_chunks_received: dict[str, dict[str, tuple]] = {}
        # Count of remote chunks whose per-layer drain has FULLY completed
        # (all 36 layers received + injected + GPU-synced). Used by
        # _maybe_launch_parent to ensure the parent doesn't launch before
        # all chunks' KV is in the cache.
        self._sage_remote_drains_done: dict[str, int] = {}
        # Parents whose drains failed — these will never launch.
        self._sage_drain_failed: set[str] = set()
        # chunk_request_id -> token_ids (for parallel prefill KV transfer).
        self._chunk_token_ids_by_req_id: dict[str, list[int]] = {}
        # Home GPU: positions we know are local for each parent.
        # parent_id -> set of positions
        self._sage_local_positions: dict[str, set[int]] = {}
        # Home GPU: remote positions we've started receiving for.
        # parent_id -> set of positions (to avoid double-starting)
        self._sage_recv_started: dict[str, set[int]] = {}
        # Home GPU 3+ GPU mode: parents auto-registered from remote chunks
        # that the frontend output_processor doesn't know about yet. These
        # are drained into the next EngineCoreOutputs in step().
        self._sage_pending_parent_announcements: list = []
        # Per-layer concurrent recv: completion events from background
        # payload threads. Items are (parent_id, chunk_req_id).
        import queue as _queue
        import threading as _threading
        self._sage_payload_done: "_queue.Queue[tuple[str, str]]" = (
            _queue.Queue()
        )
        # Per-(parent_id, layer_idx) → number of chunks that have finished
        # injecting that layer. Updated by background recv threads, read by
        # the engine core main thread to decide when to fire early blend.
        self._sage_layer_arrivals: dict[tuple[str, int], int] = {}
        self._sage_layer_arrivals_lock = _threading.Lock()
        # Concurrent prefill state tracking
        self.chunk_groups: dict[str, list[str]] = {}  # parent_req_id -> [chunk_req_ids]
        self.chunk_completion: dict[str, set[str]] = {}  # parent_req_id -> completed chunks
        self.parent_requests: dict[str, Request] = {}  # parent_req_id -> original Request
        self.chunk_lengths: dict[str, int] = {}  # chunk_req_id -> chunk token length
        self.chunk_to_parent: dict[str, str] = {}  # chunk_req_id -> parent_req_id
        self._chunk_req_to_position: dict[str, int] = {}  # chunk_req_id -> position
        # Explicit concurrent mode bookkeeping: request_id -> {chunk_id: token_ids}
        self.concurrent_chunk_payloads: dict[str, dict[int, list[int]]] = {}
        # request_id -> {position: chunk_id}
        self.concurrent_chunk_positions: dict[str, dict[int, int]] = {}
        self.concurrent_parent_template: dict[str, Request] = {}
        self.concurrent_parent_sampling_params: dict[str, Any] = {}
        self.concurrent_total_chunks: dict[str, int] = {}
        self._concurrent_query_token_count: dict[str, int] = {}
        # parent_id -> {chunk_id: [MultiModalFeatureSpec, ...]}
        self._concurrent_chunk_mm_features: dict[str, dict[int, list]] = {}
        # Track all parent_ids that have ever been submitted in this engine
        # lifetime. Reusing a parent_id causes silent parent-assembly
        # failures (chunks are accepted but the assembled parent is
        # discarded, returning empty outputs). Fail loudly instead.
        self._concurrent_consumed_parent_ids: set[str] = set()
        # parent_id -> {chunk_req_id: [[t,h,w], ...]} for remote chunks
        self._sage_remote_image_grid_thw: dict[str, dict[str, list]] = {}

        # Mark the startup heap as static so that it's ignored by GC.
        # Reduces pause times of oldest generation collections.
        freeze_gc_heap()
        # If enable, attach GC debugger after static variable freeze.
        maybe_attach_gc_debug_callback()
        # Enable environment variable cache (e.g. assume no more
        # environment variable overrides after this point)
        enable_envs_cache()

    def _initialize_kv_caches(
        self, vllm_config: VllmConfig
    ) -> tuple[int, int, KVCacheConfig]:
        start = time.time()

        # Get all kv cache needed by the model
        kv_cache_specs = self.model_executor.get_kv_cache_specs()

        has_kv_cache = any(kv_cache_spec for kv_cache_spec in kv_cache_specs)
        if has_kv_cache:
            if os.environ.get("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH") == "1":
                dp_group = getattr(self, "dp_group", None)
                assert dp_group is not None
                self.available_gpu_memory_for_kv_cache = (
                    ParallelConfig.sync_kv_cache_memory_size(dp_group, -1)
                )
                available_gpu_memory = [self.available_gpu_memory_for_kv_cache] * len(
                    kv_cache_specs
                )
            else:
                # Profiles the peak memory usage of the model to determine how
                # much memory can be allocated for kv cache.
                available_gpu_memory = self.model_executor.determine_available_memory()
                self.available_gpu_memory_for_kv_cache = available_gpu_memory[0]
        else:
            # Attention free models don't need memory for kv cache
            available_gpu_memory = [0] * len(kv_cache_specs)

        assert len(kv_cache_specs) == len(available_gpu_memory)

        kv_cache_configs = get_kv_cache_configs(
            vllm_config, kv_cache_specs, available_gpu_memory
        )
        scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
        num_gpu_blocks = scheduler_kv_cache_config.num_blocks
        num_cpu_blocks = 0

        # Initialize kv cache and warmup the execution
        self.model_executor.initialize_from_config(kv_cache_configs)

        elapsed = time.time() - start
        logger.info_once(
            "init engine (profile, create kv cache, warmup model) took %.2f seconds",
            elapsed,
            scope="local",
        )
        return num_gpu_blocks, num_cpu_blocks, scheduler_kv_cache_config

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_executor.supported_tasks

    def add_request(self, request: Request, request_wave: int = 0):
        """Add request to the scheduler.

        `request_wave`: indicate which wave of requests this is expected to
        belong to in DP case
        """
        current_time = time.time()
        logger.info("Adding request %s at time %.2f", request.request_id, current_time)

        # Validate the request_id type.
        if not isinstance(request.request_id, str):
            raise TypeError(
                f"request_id must be a string, got {type(request.request_id)}"
            )

        if pooling_params := request.pooling_params:
            supported_pooling_tasks = [
                task for task in self.get_supported_tasks() if task in POOLING_TASKS
            ]

            if pooling_params.task not in supported_pooling_tasks:
                raise ValueError(
                    f"Unsupported task: {pooling_params.task!r} "
                    f"Supported tasks: {supported_pooling_tasks}"
                )

        if request.kv_transfer_params is not None and (
            not self.scheduler.get_kv_connector()
        ):
            logger.warning(
                "Got kv_transfer_params, but no KVConnector found. "
                "Disabling KVTransfer for this request."
            )

        # Explicit request routing by request_type metadata.
        # sequential: bypass Sage and run normally.
        # concurrent: treat each input as a chunk for Sage concurrent prefill.
        if request.request_type == "sequential":
            if (
                request.parent_request_id is not None
                or request.chunk_id is not None
                or request.position is not None
                or request.total_chunks is not None
            ):
                raise ValueError(
                    "sequential requests must not include request_id "
                    "(parent_request_id), chunk_id, position, or total_chunks."
                )
            self.scheduler.add_request(request)
            return

        if request.request_type == "concurrent":
            if not self.enable_sage:
                raise ValueError(
                    "Concurrent request received but concurrent prefill "
                    "is not enabled. Set ENABLE_SAGE=True and configure "
                    "LMCache to use SAGE concurrent prefill."
                )

            if request.parent_request_id is None:
                raise ValueError(
                    "concurrent requests must provide request_id "
                    "(propagated as parent_request_id)."
                )
            if request.chunk_id is None:
                raise ValueError("concurrent requests must provide chunk_id.")
            if request.position is None:
                raise ValueError("concurrent requests must provide position.")
            if request.total_chunks is None:
                raise ValueError("concurrent requests must provide total_chunks.")
            if request.prompt_token_ids is None:
                raise ValueError(
                    "concurrent requests must provide prompt_token_ids."
                )
            if request.total_chunks <= 0:
                raise ValueError("total_chunks must be greater than 0.")
            if request.position < 0 or request.position >= request.total_chunks:
                raise ValueError(
                    "position must be in the range [0, total_chunks)."
                )

            parent_id = request.parent_request_id
            # Detect reuse of a parent_id whose parent has already been
            # assembled/launched. Without this check, subsequent chunks
            # under the same parent_id are silently accepted but the new
            # parent-assembly never reaches the scheduler, leaving the
            # caller to receive empty outputs from llm.generate().
            if (
                parent_id in self._concurrent_consumed_parent_ids
                and parent_id not in self.concurrent_chunk_payloads
            ):
                raise ValueError(
                    f"parent_request_id={parent_id!r} was already used by a "
                    "previous concurrent-prefill request. request_ids must be "
                    "unique across the engine's lifetime. (Hint: append a "
                    "per-request counter to the request_id.)"
                )
            chunk_payloads = self.concurrent_chunk_payloads.setdefault(parent_id, {})
            self._concurrent_consumed_parent_ids.add(parent_id)
            if request.chunk_id in chunk_payloads:
                raise ValueError(
                    f"Duplicate chunk_id={request.chunk_id} for "
                    f"parent_request_id={parent_id}."
                )
            chunk_payloads[request.chunk_id] = list(request.prompt_token_ids)
            # Store mm_features per chunk for multimodal support.
            if request.mm_features:
                chunk_mm = self._concurrent_chunk_mm_features.setdefault(
                    parent_id, {}
                )
                chunk_mm[request.chunk_id] = list(request.mm_features)
            chunk_positions = self.concurrent_chunk_positions.setdefault(parent_id, {})
            if request.position in chunk_positions:
                raise ValueError(
                    f"Duplicate position={request.position} for "
                    f"parent_request_id={parent_id}."
                )
            chunk_positions[request.position] = request.chunk_id
            self.concurrent_parent_template.setdefault(parent_id, deepcopy(request))
            if request.sampling_params is not None:
                self.concurrent_parent_sampling_params[parent_id] = deepcopy(
                    request.sampling_params
                )
            prev_total = self.concurrent_total_chunks.get(parent_id)
            if prev_total is None:
                self.concurrent_total_chunks[parent_id] = request.total_chunks
            elif prev_total != request.total_chunks:
                raise ValueError(
                    f"Inconsistent total_chunks for parent_request_id={parent_id}: "
                    f"got {request.total_chunks}, expected {prev_total}."
                )
            expected_total = self.concurrent_total_chunks[parent_id]
            if len(chunk_payloads) > expected_total:
                raise ValueError(
                    f"Received {len(chunk_payloads)} unique chunks for "
                    f"parent_request_id={parent_id}, exceeds total_chunks="
                    f"{expected_total}."
                )
            if len(chunk_positions) > expected_total:
                raise ValueError(
                    f"Received {len(chunk_positions)} unique positions for "
                    f"parent_request_id={parent_id}, exceeds total_chunks="
                    f"{expected_total}."
                )

            # Force each chunk request to prefill only.
            if request.sampling_params is not None:
                request.sampling_params = deepcopy(request.sampling_params)
                # Chunk requests are ingestion-only; never inherit user-level
                # min_tokens (e.g. 50) or they will keep decoding extra tokens.
                request.sampling_params.min_tokens = 0
                request.sampling_params.max_tokens = 1
                # Scheduler logic uses request.max_tokens (captured at request
                # construction), so update it as well to prevent chunk decode.
                request.max_tokens = 1

            request.is_chunk_request = True

            # Track query suffix length from whichever chunk provides it
            # (typically the last chunk).
            if request.query_token_count is not None and request.query_token_count > 0:
                self._concurrent_query_token_count[parent_id] = (
                    request.query_token_count
                )

            self.chunk_groups.setdefault(parent_id, [])
            self.chunk_completion.setdefault(parent_id, set())
            self.chunk_to_parent[request.request_id] = parent_id
            self._chunk_req_to_position[request.request_id] = request.position
            self.chunk_lengths[request.request_id] = len(request.prompt_token_ids)
            self._chunk_token_ids_by_req_id[request.request_id] = list(
                request.prompt_token_ids
            )
            if request.request_id not in self.chunk_groups[parent_id]:
                self.chunk_groups[parent_id].append(request.request_id)
            self.scheduler.register_sage_chunk(request.request_id, parent_id)
            if (
                parent_id not in self.parent_requests
                and len(chunk_payloads) == expected_total
            ):
                self.parent_requests[parent_id] = self._build_concurrent_parent_request(
                    parent_id
                )
                logger.info(
                    "[SAGE_CONCURRENT] Received all chunk payloads for parent %s "
                    "(total_chunks=%d)",
                    parent_id,
                    expected_total,
                )

            # Track local positions for parallel prefill recv logic.
            if self.sage_parallel_prefill:
                self._sage_local_positions.setdefault(
                    parent_id, set()
                ).add(request.position)

            import time as _time
            logger.info(
                "[SAGE_TIMING] chunk_submitted %s pos=%s rank=%d t=%.6f",
                request.request_id,
                request.position,
                self._sage_instance_rank,
                _time.perf_counter(),
            )

            # SAGE pipelined-blend optimization: in per-layer mode, send
            # the chunk header to the home GPU NOW (before this worker
            # has done any forward compute on the chunk). This lets the
            # home start its per-layer recv loop in parallel with the
            # worker's prefill, instead of waiting until the end. With
            # this, the early-blend trigger fires after layer 1 of the
            # actual worker prefill (~10 ms in), giving the blend the
            # full worker prefill time to overlap behind.
            if (
                self.sage_parallel_prefill
                and self._sage_per_layer_transfer
                and self._sage_instance_rank != self._sage_home_rank
                and request.home_kv_address
            ):
                num_tokens = len(request.prompt_token_ids)
                block_size = self.vllm_config.cache_config.block_size
                num_blocks = (num_tokens + block_size - 1) // block_size
                transfer_id = (
                    f"sage_{parent_id}_pos_{request.position}"
                )
                # Use the originally-requested sampling params (before
                # we forced max_tokens=1 for chunk-only execution above).
                sp = self.concurrent_parent_sampling_params.get(parent_id)
                try:
                    # Send header via SageKVTransferEngine (available
                    # on all worker instances in both bulk and per-layer).
                    header = torch.tensor(
                        [
                            num_tokens, num_blocks,
                            self.concurrent_total_chunks.get(parent_id, 0),
                            sp.max_tokens if sp else 1,
                            sp.min_tokens if sp else 0,
                            self._concurrent_query_token_count.get(
                                parent_id, 0
                            ),
                        ] + list(request.prompt_token_ids),
                        dtype=torch.long,
                        device=self._sage_kv_transfer.device,
                    )
                    self._sage_kv_transfer._nccl_engine.send_tensor(
                        f"{transfer_id}#header",
                        header,
                        request.home_kv_address,
                    )
                    logger.info(
                        "[SAGE_PARALLEL] Worker rank=%d sent EARLY header "
                        "for chunk %s (%d tokens, %d blocks, pos=%d)",
                        self._sage_instance_rank,
                        request.request_id,
                        num_tokens,
                        num_blocks,
                        request.position,
                    )
                except Exception as e:
                    logger.warning(
                        "[SAGE_PARALLEL] Early header send failed: %s",
                        e,
                    )

            if self._sage_chunk_fits(request):
                self.scheduler.add_request(request)
                logger.info(
                    "[SAGE_CONCURRENT] Added chunk request %s "
                    "(chunk_id=%s, position=%s, total_chunks=%s) "
                    "for parent %s on rank=%d",
                    request.request_id,
                    request.chunk_id,
                    request.position,
                    request.total_chunks,
                    parent_id,
                    self._sage_instance_rank,
                )
            else:
                self._sage_pending_chunks.append(request)
                logger.info(
                    "[SAGE_BACKPRESSURE] Queued chunk %s "
                    "(parent=%s, pos=%s); pending_chunks=%d "
                    "(KV too tight to admit immediately)",
                    request.request_id, parent_id, request.position,
                    len(self._sage_pending_chunks),
                )
            return

        raise ValueError(
            f"Unsupported request_type={request.request_type!r}. "
            "Expected 'sequential' or 'concurrent'."
        )

    def abort_requests(self, request_ids: list[str]):
        """Abort requests from the scheduler."""

        # TODO: The scheduler doesn't really need to know the
        # specific finish reason, TBD whether we propagate that
        # (i.e. client-aborted vs stop criteria met).
        self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)

    @contextmanager
    def log_error_detail(self, scheduler_output: SchedulerOutput):
        """Execute the model and log detailed info on failure."""
        try:
            yield
        except Exception as err:
            # We do not want to catch BaseException here since we're only
            # interested in dumping info when the exception is due to an
            # error from execute_model itself.

            # NOTE: This method is exception-free
            dump_engine_exception(
                self.vllm_config, scheduler_output, self.scheduler.make_stats()
            )
            raise err

    def _log_err_callback(self, scheduler_output: SchedulerOutput):
        """Log error details of a future that's not expected to return a result."""

        def callback(f, sched_output=scheduler_output):
            with self.log_error_detail(sched_output):
                result = f.result()
                assert result is None

        return callback

    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """Schedule, execute, and make output.

        Returns tuple of outputs and a flag indicating whether the model
        was executed.
        """

        if self._sage_pending_chunks:
            self._drain_sage_pending_chunks()
        if self._sage_pending_parents:
            self._drain_sage_pending_parents()
            
        if self.enable_sage:
            self._sage_check_deadlock()

        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        if not self.scheduler.has_requests():
            # Parallel prefill: home GPU must keep polling for remote KV
            # even when no local requests are active.
            if (
                self.sage_parallel_prefill
                and self._sage_instance_rank == self._sage_home_rank
            ):
                self._process_remote_chunk_arrivals()
                self._drain_recompute_events()
            if (
                self._sage_pending_chunks
                or self._sage_pending_parents
            ):
                self._drain_sage_pending_chunks()
                self._drain_sage_pending_parents()
                if self.scheduler.has_requests():
                    pass
                else:
                    return {}, False
            else:
                return {}, False
        # Parallel prefill: drain any recompute events that arrived
        # between the previous step end and this step start, BEFORE
        # the scheduler runs the parent's forward. Otherwise the
        # layerwise state (current_layer, selected_indices) is stale.
        if (
            self.sage_parallel_prefill
            and self._sage_instance_rank == self._sage_home_rank
        ):
            self._drain_recompute_events()

        # Stash per-chunk RoPE tasks on the connector impl BEFORE schedule()
        scheduler_output = self.scheduler.schedule()
        # Worker GPU: populate pending sends so the save_kv_layer callback
        # sends per-layer KV to home via SageKVTransferEngine during forward.
        if self._sage_per_layer_transfer:
            self._populate_pending_sends(scheduler_output)
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        with self.log_error_detail(scheduler_output):
            model_output = future.result()
            if model_output is None:
                model_output = self.model_executor.sample_tokens(grammar_output)

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )

        # Handle concurrent prefill chunk completions (now KV info is already captured)
        if self.enable_sage:
            self._process_chunk_completions(scheduler_output)

        # Parallel prefill: home GPU checks for arrived remote chunks.
        if self.sage_parallel_prefill and self._sage_instance_rank == self._sage_home_rank:
            import time as _time
            _t0 = _time.perf_counter()
            self._process_remote_chunk_arrivals()
            # Drain pipelined recompute events that arrived during
            # model execution (they sit in input_queue).
            self._drain_recompute_events()
            logger.info(
                "[SAGE_TIMING] step_end_poll t=%.6f", _time.perf_counter(),
            )

        # Drain pending parent auto-registration announcements into the
        # outputs so the frontend can register them in output_processor.
        if self._sage_pending_parent_announcements:
            from vllm.v1.engine import EngineCoreOutputs
            announcements = self._sage_pending_parent_announcements
            self._sage_pending_parent_announcements = []
            eco = engine_core_outputs.get(0)
            if eco is None:
                eco = EngineCoreOutputs()
                engine_core_outputs[0] = eco
            eco.concurrent_parents_added = announcements

        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0

    def _process_chunk_completions(self, scheduler_output) -> None:
        """Process completed chunk requests and trigger blending when ready."""
        # Get completed request IDs from scheduler output
        finished_req_ids = set(scheduler_output.finished_req_ids)

        # Filter for chunk request completions
        chunk_completions = {
            req_id for req_id in finished_req_ids
            if req_id in self.chunk_to_parent
        }

        if not chunk_completions:
            return

        # Check which parents are ready for blending
        parents_ready = self._handle_chunk_completion(chunk_completions)

        # Trigger blending for ready parents
        for parent_id in parents_ready:
            try:
                self._launch_final_request(parent_id)
            except Exception as e:
                logger.error(
                    "Failed to blend and launch final request for %s: %s",
                    parent_id, e
                )
                # Clean up state on failure
                self._cleanup_chunk_state(parent_id)

    def post_step(self, model_executed: bool) -> None:
        # When using async scheduling we can't get draft token ids in advance,
        # so we update draft token ids in the worker process and don't
        # need to update draft token ids here.
        if not self.async_scheduling and self.use_spec_decode and model_executed:
            # Take the draft token ids.
            draft_token_ids = self.model_executor.take_draft_token_ids()
            if draft_token_ids is not None:
                self.scheduler.update_draft_token_ids(draft_token_ids)

    def step_with_batch_queue(
        self,
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
        """Schedule and execute batches with the batch queue.
        Note that if nothing to output in this step, None is returned.

        The execution flow is as follows:
        1. Try to schedule a new batch if the batch queue is not full.
        If a new batch is scheduled, directly return an empty engine core
        output. In other words, fulfilling the batch queue has a higher priority
        than getting model outputs.
        2. If there is no new scheduled batch, meaning that the batch queue
        is full or no other requests can be scheduled, we block until the first
        batch in the job queue is finished.
        3. Update the scheduler from the output.
        """
        batch_queue = self.batch_queue
        assert batch_queue is not None

        # Try to schedule a new batch if the batch queue is not full, but
        # the scheduler may return an empty batch if all requests are scheduled.
        # Note that this is not blocking.
        assert len(batch_queue) < self.batch_queue_size

        model_executed = False
        deferred_scheduler_output = None
        if self.scheduler.has_requests():
            scheduler_output = self.scheduler.schedule()
            if self._sage_per_layer_transfer:
                self._populate_pending_sends(scheduler_output)
            exec_future = self.model_executor.execute_model(
                scheduler_output, non_block=True
            )
            if not self.is_ec_producer:
                model_executed = scheduler_output.total_num_scheduled_tokens > 0

            if self.is_pooling_model or not model_executed:
                # No sampling required (no requests scheduled).
                future = cast(Future[ModelRunnerOutput], exec_future)
            else:
                exec_future.add_done_callback(self._log_err_callback(scheduler_output))

                if not scheduler_output.pending_structured_output_tokens:
                    # We aren't waiting for any tokens, get any grammar output
                    # and sample immediately.
                    grammar_output = self.scheduler.get_grammar_bitmask(
                        scheduler_output
                    )
                    future = self.model_executor.sample_tokens(
                        grammar_output, non_block=True
                    )
                else:
                    # We need to defer sampling until we have processed the model output
                    # from the prior step.
                    deferred_scheduler_output = scheduler_output

            if not deferred_scheduler_output:
                # Add this step's future to the queue.
                batch_queue.appendleft((future, scheduler_output))
                if (
                    model_executed
                    and len(batch_queue) < self.batch_queue_size
                    and not batch_queue[-1][0].done()
                ):
                    # Don't block on next worker response unless the queue is full
                    # or there are no more requests to schedule.
                    return None, True

        elif not batch_queue:
            # Queue is empty. We should not reach here since this method should
            # only be called when the scheduler contains requests or the queue
            # is non-empty.
            return None, False

        # Block until the next result is available.
        future, scheduler_output = batch_queue.pop()
        with self.log_error_detail(scheduler_output):
            model_output = future.result()

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )

        # NOTE(nick): We can either handle the deferred tasks here or save
        # in a field and do it immediately once step_with_batch_queue is
        # re-called. The latter slightly favors TTFT over TPOT/throughput.
        if deferred_scheduler_output:
            # We now have the tokens needed to compute the bitmask for the
            # deferred request. Get the bitmask and call sample tokens.
            grammar_output = self.scheduler.get_grammar_bitmask(
                deferred_scheduler_output
            )
            future = self.model_executor.sample_tokens(grammar_output, non_block=True)
            batch_queue.appendleft((future, deferred_scheduler_output))

        return engine_core_outputs, model_executed

    def _process_aborts_queue(self):
        if not self.aborts_queue.empty():
            request_ids = []
            while not self.aborts_queue.empty():
                ids = self.aborts_queue.get_nowait()
                if isinstance(ids, str):
                    # Should be a list here, but also handle string just in case.
                    ids = (ids,)
                request_ids.extend(ids)
            # More efficient to abort all as a single batch.
            self.abort_requests(request_ids)

    def shutdown(self):
        self.structured_output_manager.clear_backend()
        if self.model_executor:
            self.model_executor.shutdown()
        if self.scheduler:
            self.scheduler.shutdown()

    def profile(self, is_start: bool = True):
        self.model_executor.profile(is_start)

    def reset_mm_cache(self):
        # NOTE: Since this is mainly for debugging, we don't attempt to
        # re-sync the internal caches (P0 sender, P1 receiver)
        if self.scheduler.has_unfinished_requests():
            logger.warning(
                "Resetting the multi-modal cache when requests are "
                "in progress may lead to desynced internal caches."
            )

        # The cache either exists in EngineCore or WorkerWrapperBase
        if self.mm_receiver_cache is not None:
            self.mm_receiver_cache.clear_cache()

        self.model_executor.reset_mm_cache()

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return self.scheduler.reset_prefix_cache(
            reset_running_requests, reset_connector
        )

    def sleep(self, level: int = 1):
        self.model_executor.sleep(level)

    def wake_up(self, tags: list[str] | None = None):
        self.model_executor.wake_up(tags)

    def is_sleeping(self) -> bool:
        return self.model_executor.is_sleeping

    def execute_dummy_batch(self):
        self.model_executor.execute_dummy_batch()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_executor.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_executor.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_executor.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_executor.pin_lora(lora_id)

    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        self.model_executor.save_sharded_state(
            path=path, pattern=pattern, max_size=max_size
        )

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return self.model_executor.collective_rpc(method, timeout, args, kwargs)

    # ==================== Concurrent Prefill Methods ====================

    def _build_concurrent_parent_request(self, parent_id: str) -> Request:
        """Build the final parent request from received concurrent chunks."""
        if parent_id not in self.concurrent_parent_template:
            raise ValueError(f"Missing parent template for request_id={parent_id}")

        chunk_payloads = self.concurrent_chunk_payloads.get(parent_id, {})
        if not chunk_payloads:
            raise ValueError(f"No chunk payloads found for request_id={parent_id}")
        chunk_positions = self.concurrent_chunk_positions.get(parent_id, {})
        if len(chunk_positions) != len(chunk_payloads):
            raise ValueError(
                "Mismatch between chunk positions and payloads for "
                f"request_id={parent_id}: positions={len(chunk_positions)} "
                f"payloads={len(chunk_payloads)}."
            )
        expected_total = self.concurrent_total_chunks.get(parent_id)
        if expected_total is None:
            raise ValueError(f"Missing total_chunks for request_id={parent_id}")
        if len(chunk_positions) != expected_total:
            raise ValueError(
                f"Incomplete chunk positions for request_id={parent_id}: "
                f"positions={len(chunk_positions)} expected={expected_total}."
            )
        expected_positions = set(range(expected_total))
        actual_positions = set(chunk_positions)
        if actual_positions != expected_positions:
            raise ValueError(
                f"Invalid chunk positions for request_id={parent_id}: "
                f"got={sorted(actual_positions)} expected={sorted(expected_positions)}."
            )

        template_request = self.concurrent_parent_template[parent_id]
        ordered_chunk_ids = [chunk_positions[pos] for pos in range(expected_total)]
        logger.info(
            "[SAGE_CONCURRENT] Assembling parent %s in position order %s",
            parent_id,
            ordered_chunk_ids,
        )
        full_tokens: list[int] = []
        chunk_boundaries: list[int] = []
        position = 0
        for chunk_id in ordered_chunk_ids:
            chunk_boundaries.append(position)
            chunk_tokens = chunk_payloads[chunk_id]
            full_tokens.extend(chunk_tokens)
            position += len(chunk_tokens)

        sampling_params = self.concurrent_parent_sampling_params.get(parent_id)
        if sampling_params is not None:
            sampling_params = deepcopy(sampling_params)
        elif template_request.sampling_params is not None:
            sampling_params = deepcopy(template_request.sampling_params)

        pooling_params = (
            deepcopy(template_request.pooling_params)
            if template_request.pooling_params is not None
            else None
        )

        # Merge mm_features from chunks in position order,
        # adjusting mm_position offsets for the merged token sequence.
        # For remote chunks, reconstruct mm_features from received
        # image_grid_thw metadata.
        merged_mm_features = None
        chunk_mm = self._concurrent_chunk_mm_features.get(parent_id)
        remote_grid_thw = self._sage_remote_image_grid_thw.get(parent_id, {})
        has_any_mm = bool(chunk_mm) or bool(remote_grid_thw)
        if has_any_mm:
            from vllm.multimodal.inputs import (
                MultiModalFeatureSpec, MultiModalFieldElem,
                MultiModalBatchedField, PlaceholderRange,
            )
            import torch as _torch
            chunk_mm = chunk_mm or {}
            merged_mm_features = []
            # Get a reference feature from any local chunk for template data
            _ref_feat = None
            for feats in chunk_mm.values():
                if feats:
                    _ref_feat = feats[0]
                    break
            token_offset = 0
            for chunk_id in ordered_chunk_ids:
                chunk_tokens = chunk_payloads.get(chunk_id, [])
                if chunk_id in chunk_mm:
                    # Local chunk: use stored mm_features with offset adjustment
                    for feat in chunk_mm[chunk_id]:
                        if feat.mm_position is not None and token_offset > 0:
                            new_pos = PlaceholderRange(
                                offset=feat.mm_position.offset + token_offset,
                                length=feat.mm_position.length,
                                is_embed=feat.mm_position.is_embed,
                            )
                            adjusted = MultiModalFeatureSpec(
                                data=feat.data,
                                modality=feat.modality,
                                identifier=feat.identifier,
                                mm_position=new_pos,
                            )
                            merged_mm_features.append(adjusted)
                        else:
                            merged_mm_features.append(feat)
                elif chunk_id in remote_grid_thw and _ref_feat is not None:
                    # Remote chunk: reconstruct from image_grid_thw.
                    # Scan tokens for vision_start to find image positions.
                    grid_thws = remote_grid_thw[chunk_id]
                    vision_start_id = 151652  # <|vision_start|>
                    img_idx = 0
                    for t_idx, tok in enumerate(chunk_tokens):
                        if tok == vision_start_id and img_idx < len(grid_thws):
                            thw = grid_thws[img_idx]
                            # Compute placeholder length from grid dimensions
                            merge_size = 2  # spatial_merge_size default
                            length = (thw[0] * thw[1] * thw[2])
                            length = length // (merge_size * merge_size)
                            data = {
                                "image_grid_thw": MultiModalFieldElem(
                                    modality="image",
                                    key="image_grid_thw",
                                    data=_torch.tensor(thw),
                                    field=MultiModalBatchedField(
                                        keep_on_cpu=True,
                                    ),
                                ),
                            }
                            new_pos = PlaceholderRange(
                                offset=token_offset + t_idx + 1,
                                length=length,
                            )
                            merged_mm_features.append(
                                MultiModalFeatureSpec(
                                    data=data,
                                    modality="image",
                                    identifier=f"remote_{chunk_id}_{img_idx}",
                                    mm_position=new_pos,
                                )
                            )
                            img_idx += 1
                token_offset += len(chunk_tokens)

        parent_request = Request(
            request_id=parent_id,
            prompt_token_ids=full_tokens,
            mm_features=merged_mm_features,
            sampling_params=sampling_params,
            pooling_params=pooling_params,
            eos_token_id=template_request.eos_token_id,
            client_index=template_request.client_index,
            arrival_time=template_request.arrival_time,
            lora_request=template_request.lora_request,
            cache_salt=template_request.cache_salt,
            priority=template_request.priority,
            trace_headers=template_request.trace_headers,
            block_hasher=self.request_block_hasher,
            request_type="concurrent",
            parent_request_id=parent_id,
            total_chunks=self.concurrent_total_chunks.get(parent_id),
            query_token_count=self._concurrent_query_token_count.get(parent_id, 0),
        )
        parent_request.sage_chunk_boundaries = chunk_boundaries
        return parent_request

    def _handle_chunk_completion(self, completed_req_ids: set[str]) -> list[str]:
        """Check for completed chunks and trigger blending when all chunks done.

        In single-GPU mode: returns parent IDs that are ready for blending
        when all chunks complete locally.

        In parallel prefill mode:
        - Worker GPU (not home): sends KV blocks to home GPU via NCCL,
          does NOT add to parents_ready (home GPU handles assembly).
        - Home GPU: tracks local chunk completions. Parent is only ready
          after all local chunks AND all remote KV transfers complete.
          Remote completions are registered via receive_remote_chunk_kv().
        """
        parents_ready_for_blending = []

        import time as _time
        for req_id in completed_req_ids:
            if req_id not in self.chunk_to_parent:
                continue
            parent_id = self.chunk_to_parent[req_id]

            logger.info(
                "[SAGE_TIMING] chunk_done %s parent=%s rank=%d t=%.6f",
                req_id, parent_id, self._sage_instance_rank,
                _time.perf_counter(),
            )

            # Mark chunk as complete locally.
            self.chunk_completion[parent_id].add(req_id)

            # ── Parallel prefill: worker GPU sends KV to home ──
            if (
                self.sage_parallel_prefill
                and self._sage_instance_rank != self._sage_home_rank
            ):
                if self._sage_per_layer_transfer:
                    # Header was already sent EARLY in
                    # _handle_concurrent_request to enable pipelined
                    # blend overlap. Nothing more to do here — the
                    # per-layer KV streamed during forward via
                    # SageKVTransferEngine callback and is already in
                    # home's recv_store ready to be drained.
                    self.scheduler.free_sage_chunk_blocks(
                        parent_id, chunk_req_id=req_id,
                    )
                else:
                    self._send_chunk_kv_to_home(req_id, parent_id)
                continue

            # ── Single-GPU or home GPU: check if all chunks are done ──
            if self.sage_parallel_prefill:
                self._maybe_launch_parent(parent_id)
            else:
                # Single-GPU: all chunks are local.
                expected_total = self.concurrent_total_chunks.get(parent_id)
                if expected_total is None:
                    raise RuntimeError(
                        f"Missing total_chunks for parent {parent_id} "
                        f"while processing chunk {req_id}"
                    )
                total_done = len(self.chunk_completion[parent_id])
                if total_done == expected_total:
                    if parent_id not in self.parent_requests:
                        self.parent_requests[parent_id] = (
                            self._build_concurrent_parent_request(parent_id)
                        )
                    parents_ready_for_blending.append(parent_id)
                    logger.info(
                        "Completed %d/%d chunks for parent %s, "
                        "ready for blending",
                        total_done,
                        expected_total,
                        parent_id,
                    )
                elif total_done > expected_total:
                    raise RuntimeError(
                        f"Completed chunks ({total_done}) exceeded "
                        f"total_chunks ({expected_total}) for "
                        f"parent {parent_id}"
                    )

        return parents_ready_for_blending

    def _send_chunk_kv_to_home(self, chunk_req_id: str, parent_id: str) -> None:
        """Worker GPU (bulk mode): extract KV blocks for a completed chunk
        and send to the home GPU via SageKVTransferEngine.
        """
        if self._sage_kv_transfer is None:
            raise RuntimeError(
                "SAGE parallel prefill: KV transfer engine not initialized "
                f"on dp_rank={self._sage_instance_rank}"
            )
        home_addr = getattr(
            self.concurrent_parent_template.get(parent_id),
            "home_kv_address",
            None,
        )
        if not home_addr:
            raise RuntimeError(
                f"No home_kv_address for parent {parent_id} "
                f"on dp_rank={self._sage_instance_rank}"
            )
        chunk_block_info = self.scheduler.sage_parent_chunk_blocks.get(
            parent_id, []
        )
        block_ids: list[int] = []
        num_tokens = 0
        position = -1
        for cid, raw_block_ids, ntokens, _pos in chunk_block_info:
            if cid == chunk_req_id:
                block_ids = [
                    int(blk.block_id) if hasattr(blk, "block_id") else int(blk)
                    for blk in raw_block_ids[0]
                ] if raw_block_ids else []
                num_tokens = ntokens
                position = _pos
                break
        if not block_ids:
            raise RuntimeError(
                f"No block IDs found for chunk {chunk_req_id} "
                f"(parent {parent_id}) on dp_rank={self._sage_instance_rank}"
            )
        token_ids = self._chunk_token_ids_by_req_id.get(chunk_req_id)
        if not token_ids:
            raise RuntimeError(
                f"No token_ids found for chunk {chunk_req_id} "
                f"(parent {parent_id}) on dp_rank={self._sage_instance_rank}"
            )
        transfer_id = f"sage_{parent_id}_pos_{position}"
        # Extract image_grid_thw for multimodal support.
        # Remote home needs this to compute M-RoPE positions.
        image_grid_thw = []
        chunk_mm = self._concurrent_chunk_mm_features.get(parent_id, {})
        chunk_feats = chunk_mm.get(
            self.concurrent_chunk_positions.get(parent_id, {}).get(position)
        )
        if chunk_feats:
            from vllm.multimodal.inputs import MultiModalFeatureSpec
            kwargs = MultiModalFeatureSpec.gather_kwargs(
                chunk_feats, {"image_grid_thw"},
            )
            image_grid_thw = [
                item.tolist() for item in kwargs.get("image_grid_thw", [])
            ]
        logger.info(
            "[SAGE_PARALLEL] Worker rank=%d sending chunk %s "
            "(%d blocks, %d tokens, %d images) → %s",
            self._sage_instance_rank,
            chunk_req_id,
            len(block_ids),
            num_tokens,
            len(image_grid_thw),
            home_addr,
        )
        # Get generation params from parent template for home registration.
        template = self.concurrent_parent_template.get(parent_id)
        sp = self.concurrent_parent_sampling_params.get(parent_id)
        self._sage_kv_transfer.send_chunk(
            block_ids=block_ids,
            token_ids=token_ids,
            dest_address=home_addr,
            transfer_id=transfer_id,
            image_grid_thw=image_grid_thw,
            total_chunks=self.concurrent_total_chunks.get(parent_id, 0),
            max_tokens=sp.max_tokens if sp else 1,
            min_tokens=sp.min_tokens if sp else 0,
            query_token_count=self._concurrent_query_token_count.get(
                parent_id, 0
            ),
        )
        self.scheduler.free_sage_chunk_blocks(
            parent_id, chunk_req_id=chunk_req_id,
        )

    def _populate_pending_sends(self, scheduler_output) -> None:
        """Worker GPU: populate SageKVTransferEngine._pending_sends so
        the save_kv_layer callback sends per-layer KV during forward.

        Iterates scheduled requests, finds SAGE chunks, and builds
        send info (transfer_id, block_ids tensor, dest_address).
        """
        if self._sage_kv_transfer is None:
            return
        pending = []
        # Check new requests scheduled this step.
        for new_req in scheduler_output.scheduled_new_reqs:
            parent_id = self.chunk_to_parent.get(new_req.req_id)
            if parent_id is None:
                continue
            position = self._chunk_req_to_position.get(new_req.req_id)
            if position is None:
                continue
            template = self.concurrent_parent_template.get(parent_id)
            if template is None or not template.home_kv_address:
                continue
            transfer_id = f"sage_{parent_id}_pos_{position}"
            block_ids = new_req.block_ids[0] if new_req.block_ids else []
            block_ids_t = torch.tensor(
                block_ids, dtype=torch.long,
                device=self._sage_kv_transfer.device,
            )
            pending.append({
                "transfer_id": transfer_id,
                "block_ids": block_ids_t,
                "dest_address": template.home_kv_address,
            })
            logger.info(
                "[SAGE_PER_LAYER_SEND] rank=%d queued %s -> %s "
                "(%d blocks) dest=%s",
                self._sage_instance_rank, new_req.req_id,
                transfer_id, len(block_ids),
                template.home_kv_address,
            )
        # Also check cached requests (chunked prefill continuation).
        cached = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached.req_ids):
            parent_id = self.chunk_to_parent.get(req_id)
            if parent_id is None:
                continue
            position = self._chunk_req_to_position.get(req_id)
            if position is None:
                continue
            template = self.concurrent_parent_template.get(parent_id)
            if template is None or not template.home_kv_address:
                continue
            transfer_id = f"sage_{parent_id}_pos_{position}"
            new_block_ids = cached.new_block_ids[i]
            if new_block_ids is not None:
                # For continued chunked prefill, accumulate all blocks.
                block_ids = new_block_ids[0]
            else:
                block_ids = []
            if not block_ids:
                continue
            block_ids_t = torch.tensor(
                block_ids, dtype=torch.long,
                device=self._sage_kv_transfer.device,
            )
            pending.append({
                "transfer_id": transfer_id,
                "block_ids": block_ids_t,
                "dest_address": template.home_kv_address,
            })
        self._sage_kv_transfer._pending_sends = pending

    def _receive_remote_chunk(
        self,
        parent_id: str,
        position: int,
    ) -> None:
        """Home GPU: receive one remote chunk's header + per-layer KV."""
        if self._sage_kv_transfer is None:
            raise RuntimeError(
                "SAGE parallel prefill: KV transfer engine not initialized "
                f"on home dp_rank={self._sage_instance_rank}"
            )

        transfer_id = f"sage_{parent_id}_pos_{position}"
        chunk_req_id = f"{parent_id}_remote_pos_{position}"

        import time as _time
        _t_recv_start = _time.perf_counter()
        logger.info(
            "[SAGE_TIMING] recv_start pos=%d t=%.6f",
            position, _t_recv_start,
        )

        def _allocate(num_blocks: int) -> list[int]:
            return self.scheduler.allocate_sage_receive_blocks(
                parent_id=parent_id,
                chunk_req_id=chunk_req_id,
                num_blocks=num_blocks,
                num_tokens=0,
                position=position,
            )

        image_grid_thw: list[list[int]] = []
        payload_thread = None
        if self._sage_per_layer_transfer:
            # Step 1: receive the (small) header on the main thread.
            # This unblocks alloc + state registration immediately.
            meta = self._sage_kv_transfer.recv_chunk_per_layer_header(
                transfer_id
            )
            dest_block_ids = _allocate(meta["num_blocks"])
            result = {
                "token_ids": meta["token_ids"],
                "dest_block_ids": dest_block_ids,
                "num_tokens": meta["num_tokens"],
                "num_blocks": meta["num_blocks"],
                "image_grid_thw": [],
                "total_chunks": meta["total_chunks"],
                "max_tokens": meta["max_tokens"],
                "min_tokens": meta["min_tokens"],
                "query_token_count": meta["query_token_count"],
            }
            # Step 2: spawn a background thread to drain the per-layer
            # payload. Multiple chunks can be draining concurrently —
            # this is the whole point of per-layer mode on slow links.
            import threading as _threading
            done_queue = self._sage_payload_done
            input_queue = getattr(self, "input_queue", None)
            arrivals = self._sage_layer_arrivals
            arrivals_lock = self._sage_layer_arrivals_lock
            # Number of blend layers (configurable via env, default 2).
            # We watch arrivals up to this many layers and trigger early
            # blend when all chunks have completed all blend layers.
            pre_ttft_ratio = float(
                os.environ.get("SAGE_LAYERWISE_PRE_TTFT_RATIO", "0.0")
            )
            num_layers = self._sage_kv_transfer._num_layers
            _strategy_env = os.environ.get(
                "LMCACHE_BLEND_INCREMENTAL_STRATEGY", "layer_wise"
            )
            if os.environ.get("SAGE_DISABLE_EARLY_BLEND", "0") == "1":
                blend_layers_count = 0  # disable all pipelined recompute
            elif _strategy_env == "layer_wise":
                # Layerwise: only process ptt fraction of layers during
                # drain. Remaining layers use fused inject during decode.
                blend_layers_count = max(2, int(num_layers * pre_ttft_ratio))
            else:
                # Sync cacheblend / tokenwise: process all layers during
                # drain so sage_process_layer handles everything and
                # early_blend_done prevents double processing.
                blend_layers_count = num_layers

            # Read expected_chunks at call time (not closure time)
            # because concurrent_total_chunks may not be set yet when
            # the first chunk's drain thread is created.
            _total_chunks_dict = self.concurrent_total_chunks

            def _on_layer_done(L, pid=parent_id):
                expected = _total_chunks_dict.get(pid, 0)
                with arrivals_lock:
                    arrivals[(pid, L)] = arrivals.get((pid, L), 0) + 1
                    count = arrivals.get((pid, L), 0)
                    layer_ready = (count == expected and expected > 0)
                if not layer_ready:
                    return
                if L < blend_layers_count:
                    # Pipelined recompute: process this layer
                    # (L0-L1 scoring, L2+ M-token recompute).
                    if input_queue is not None:
                        try:
                            input_queue.put_nowait(
                                ("_SAGE_RECOMPUTE_LAYER", pid, L)
                            )
                        except Exception:
                            pass

            def _drain_payload(
                tid=transfer_id,
                dest=dest_block_ids,
                pid_local=parent_id,
                cid_local=chunk_req_id,
            ):
                _error = None
                try:
                    self._sage_kv_transfer.recv_chunk_per_layer_payload(
                        tid, dest, on_layer_done=_on_layer_done,
                    )
                except Exception as exc:
                    logger.error(
                        "[SAGE_PARALLEL] Drain FAILED for %s chunk %s",
                        pid_local, cid_local, exc_info=True,
                    )
                    _error = exc
                # Always signal done (with optional error) so the parent
                # doesn't hang forever.
                done_queue.put((pid_local, cid_local, _error))
                # Wake the main busy loop so it drains the done queue.
                if input_queue is not None:
                    try:
                        input_queue.put_nowait(("_SAGE_REMOTE_KV",))
                    except Exception:
                        pass

            payload_thread = _threading.Thread(
                target=_drain_payload,
                name=f"sage-recv-{transfer_id}",
                daemon=True,
            )
            payload_thread.start()
        else:
            result = self._sage_kv_transfer.recv_chunk(
                transfer_id=transfer_id,
                allocate_blocks_fn=_allocate,
            )
        token_ids = result["token_ids"]
        dest_block_ids = result["dest_block_ids"]
        num_tokens = result["num_tokens"]
        num_blocks = result["num_blocks"]
        image_grid_thw = result["image_grid_thw"]

        # Auto-register parent if home has no local chunks (3+ GPU mode).
        # The first remote chunk carries enough metadata to set up
        # the parent tracking that normally happens when a local
        # chunk is submitted via _handle_concurrent_request.
        if parent_id not in self.concurrent_total_chunks:
            total = result["total_chunks"]
            if total > 0:
                self.concurrent_total_chunks[parent_id] = total
                self.chunk_completion.setdefault(parent_id, set())
                self.chunk_groups.setdefault(parent_id, [])
                self._concurrent_query_token_count[parent_id] = (
                    result["query_token_count"]
                )
                from vllm.sampling_params import SamplingParams
                sp = SamplingParams(
                    temperature=0.0,
                    max_tokens=result["max_tokens"],
                    min_tokens=result["min_tokens"],
                )
                self.concurrent_parent_sampling_params[parent_id] = sp
                # Create a minimal template Request for parent assembly.
                from vllm.v1.request import Request
                template = Request(
                    request_id=parent_id,
                    prompt_token_ids=token_ids,
                    sampling_params=sp,
                    pooling_params=None,
                    eos_token_id=getattr(
                        self.vllm_config.model_config.hf_config,
                        "eos_token_id",
                        None,
                    ),
                    arrival_time=_time.perf_counter(),
                    lora_request=None,
                    request_type="concurrent",
                    parent_request_id=parent_id,
                    total_chunks=total,
                )
                self.concurrent_parent_template[parent_id] = template
                # Notify the frontend so output_processor can track
                # this parent. Sent on next step() in EngineCoreOutputs.
                from vllm.v1.engine import EngineCoreRequest
                self._sage_pending_parent_announcements.append(
                    EngineCoreRequest(
                        request_id=parent_id,
                        prompt_token_ids=[],
                        mm_features=None,
                        sampling_params=sp,
                        pooling_params=None,
                        eos_token_id=(
                            lambda _eid: _eid[0] if isinstance(_eid, list) else _eid
                        )(getattr(
                            self.vllm_config.model_config.hf_config,
                            "eos_token_id",
                            None,
                        )),
                        arrival_time=_time.perf_counter(),
                        lora_request=None,
                        cache_salt=None,
                        data_parallel_rank=None,
                        request_type="concurrent",
                        parent_request_id=parent_id,
                        total_chunks=total,
                    )
                )
                logger.info(
                    "[SAGE_PARALLEL] Auto-registered parent %s from "
                    "remote chunk (total_chunks=%d, max_tokens=%d)",
                    parent_id, total, result["max_tokens"],
                )

        # Update num_tokens in sage_parent_chunk_blocks.
        for i, entry in enumerate(
            self.scheduler.sage_parent_chunk_blocks.get(parent_id, [])
        ):
            if entry[0] == chunk_req_id:
                self.scheduler.sage_parent_chunk_blocks[parent_id][i] = (
                    entry[0], entry[1], num_tokens, entry[3]
                )
                break

        chunk_payloads = self.concurrent_chunk_payloads.setdefault(parent_id, {})
        chunk_payloads[chunk_req_id] = token_ids
        chunk_positions = self.concurrent_chunk_positions.setdefault(parent_id, {})
        chunk_positions[position] = chunk_req_id
        self.chunk_lengths[chunk_req_id] = num_tokens

        remote = self._sage_remote_chunks_received.setdefault(parent_id, {})
        remote[chunk_req_id] = (position, dest_block_ids, num_tokens)
        if image_grid_thw:
            remote_mm = self._sage_remote_image_grid_thw.setdefault(parent_id, {})
            remote_mm[chunk_req_id] = image_grid_thw
        logger.info(
            "[SAGE_TIMING] recv_done pos=%d t=%.6f elapsed=%.3fms",
            position, _time.perf_counter(),
            (_time.perf_counter() - _t_recv_start) * 1000,
        )

        # In per-layer mode the actual KV is still being drained by the
        # background thread; the launch happens later when the thread
        # signals completion via _sage_payload_done. In bulk mode all
        # KV is already in place, so we can launch immediately.
        if payload_thread is None:
            # Bulk mode: KV is fully received, count as drain-done.
            self._sage_remote_drains_done[parent_id] = (
                self._sage_remote_drains_done.get(parent_id, 0) + 1
            )
            self._maybe_launch_parent(parent_id)

    def _drain_recompute_events(self) -> None:
        """Process any pending _SAGE_RECOMPUTE_LAYER events from the
        input queue. Called during step() so recompute events that
        arrive while the model is executing get processed promptly.
        """
        _input_q = getattr(self, "input_queue", None)
        if _input_q is None:
            return
        requeued = []
        while not _input_q.empty():
            try:
                req = _input_q.get_nowait()
            except Exception:
                break
            if (
                isinstance(req, tuple) and len(req) >= 3
                and req[0] == "_SAGE_RECOMPUTE_LAYER"
            ):
                self._sage_run_pipelined_recompute(req[1], req[2])
            else:
                # Not a recompute event — put it back.
                requeued.append(req)
        for item in requeued:
            _input_q.put_nowait(item)

    def _sage_run_pipelined_recompute(
        self, parent_id: str, layer_idx: int,
    ) -> None:
        """Process a single layer for pipelined scoring/recompute.

        Called from the main loop when layer_idx has been drained from
        all chunks. For L0-L1: scoring (diff-k). For L2+: M-token
        recompute. All layers: RoPE correction first.
        """
        if parent_id in self._sage_drain_failed:
            return
        # On first layer, set up cached prompt inputs for the adapter.
        if layer_idx == 0:
            remote_info = self._sage_remote_chunks_received.get(parent_id, {})
            if not remote_info:
                return
            # Gather all chunk token_ids and block_ids in position order.
            chunk_payloads = self.concurrent_chunk_payloads.get(parent_id, {})
            chunk_positions = self.concurrent_chunk_positions.get(parent_id, {})
            all_token_ids: list[int] = []
            all_block_ids: list[int] = []
            chunk_boundaries: list[int] = [0]
            for pos in sorted(chunk_positions.keys()):
                cid = chunk_positions[pos]
                tids = chunk_payloads.get(cid, [])
                all_token_ids.extend(tids)
                chunk_boundaries.append(len(all_token_ids))
                # Get block_ids for this position.
                if cid in remote_info:
                    _, dest_blocks, _ = remote_info[cid]
                    all_block_ids.extend(dest_blocks)
            if not all_token_ids:
                return
            N = len(all_token_ids)
            device = self._sage_kv_transfer.device
            block_size = 16  # TODO: get from config
            token_ids_t = torch.tensor(
                all_token_ids, dtype=torch.long, device=device,
            )
            block_ids_t = torch.tensor(
                all_block_ids, dtype=torch.long, device=device,
            )
            positions = torch.arange(N, device=device, dtype=torch.long)
            blend_slot_mapping = (
                block_ids_t[positions // block_size] * block_size
                + positions % block_size
            )
            # Store for the adapter.
            _query_tc = self._concurrent_query_token_count.get(parent_id, 0)
            self.collective_rpc(
                "sage_setup_pipelined_prompt",
                args=(
                    parent_id, token_ids_t, blend_slot_mapping,
                    chunk_boundaries, N, _query_tc,
                ),
            )

        try:
            self.collective_rpc(
                "sage_run_pipelined_recompute_layer",
                args=(parent_id, layer_idx),
            )
        except Exception as e:
            logger.warning(
                "[SAGE_RECOMPUTE] Layer %d failed for %s: %s",
                layer_idx, parent_id, e,
            )
            return

    def _process_remote_chunk_arrivals(self) -> None:
        """Home GPU: check if any remote chunks have data in recv_store.

        For each parent we know about, check all remote positions.
        Also scan recv_store for unregistered parents (3+ GPU mode
        where home has no local chunks).

        In per-layer mode, the actual KV draining happens on background
        threads. We only need a chunk's #header to be present to start
        receiving — multiple chunks can drain in parallel.
        """
        if self._sage_kv_transfer is None:
            return
        nccl_engine = self._sage_kv_transfer._nccl_engine

        # Drain payload-completion notifications from background recv
        # threads (per-layer mode). Each item means one chunk has been
        # fully received into its destination blocks.
        while True:
            try:
                item = self._sage_payload_done.get_nowait()
            except Exception:
                break
            pid_done, _cid_done = item[0], item[1]
            _drain_err = item[2] if len(item) > 2 else None
            if _drain_err is not None:
                logger.error(
                    "[SAGE_PARALLEL] Aborting parent %s: drain failed "
                    "for chunk %s: %s", pid_done, _cid_done, _drain_err,
                )
                self._sage_drain_failed.add(pid_done)
                continue
            # Track actual drain completions (not just header arrivals)
            self._sage_remote_drains_done[pid_done] = (
                self._sage_remote_drains_done.get(pid_done, 0) + 1
            )
            logger.info(
                "[SAGE_PARALLEL] Drain done for %s chunk %s "
                "(%d/%d remote drains complete)",
                pid_done, _cid_done,
                self._sage_remote_drains_done[pid_done],
                len(self._sage_remote_chunks_received.get(pid_done, {})),
            )
            self._maybe_launch_parent(pid_done)

        # In per-layer mode, the readiness sentinel is just #header
        # (the payload thread will block until all layers arrive).
        # In bulk mode, we need #metadata to know what to recv.
        if self._sage_per_layer_transfer:
            sentinel_suffix = "#header"
        else:
            sentinel_suffix = "#metadata"

        # Discover unregistered parents from recv_store keys (3+ GPU mode
        # where home has no local chunks). Scan for headers from unknown
        # parents and process them all in one pass — per-layer mode can
        # start multiple background drains in parallel.
        unregistered: list[tuple[str, int]] = []
        with nccl_engine.recv_store_cv:
            for key in nccl_engine.recv_store:
                if not key.endswith(sentinel_suffix):
                    continue
                prefix = key[: -len(sentinel_suffix)]
                parts = prefix.rsplit("_pos_", 1)
                if len(parts) != 2:
                    continue
                pid = parts[0][len("sage_"):]
                pos = int(parts[1])
                if pid in self.concurrent_total_chunks:
                    continue  # already registered
                if self._sage_per_layer_transfer:
                    unregistered.append((pid, pos))
                else:
                    bulk_key = f"{prefix}#bulk_kv"
                    if bulk_key in nccl_engine.recv_store:
                        unregistered.append((pid, pos))
        for pid, pos in unregistered:
            started = self._sage_recv_started.setdefault(pid, set())
            if pos in started:
                continue
            started.add(pos)
            self._receive_remote_chunk(pid, pos)

        for parent_id, total in list(self.concurrent_total_chunks.items()):
            local_pos = self._sage_local_positions.get(parent_id, set())
            started = self._sage_recv_started.setdefault(parent_id, set())

            for pos in range(total):
                if pos in local_pos or pos in started:
                    continue
                transfer_id = f"sage_{parent_id}_pos_{pos}"
                with nccl_engine.recv_store_cv:
                    if self._sage_per_layer_transfer:
                        ready = (
                            f"{transfer_id}#header"
                            in nccl_engine.recv_store
                        )
                    else:
                        ready = (
                            f"{transfer_id}#metadata"
                            in nccl_engine.recv_store
                            and f"{transfer_id}#bulk_kv"
                            in nccl_engine.recv_store
                        )
                if not ready:
                    continue
                started.add(pos)
                self._receive_remote_chunk(parent_id, pos)

    def _maybe_launch_parent(self, parent_id: str) -> None:
        """Check if all chunks (local + remote) are done and launch parent."""
        if parent_id in self._sage_drain_failed:
            logger.error(
                "[SAGE_PARALLEL] Parent %s: drain failed, aborting",
                parent_id,
            )
            self._partial_cleanup_chunk_state(parent_id)
            self._sage_drain_failed.discard(parent_id)
            return
        expected_total = self.concurrent_total_chunks.get(parent_id)
        if expected_total is None:
            return
        local_done = len(self.chunk_completion.get(parent_id, set()))
        # Use drain completion count, NOT header arrival count.
        # Headers arrive almost instantly; drains take time to inject
        # all 36 layers of KV into the cache.
        remote_done = self._sage_remote_drains_done.get(parent_id, 0)
        total_done = local_done + remote_done
        logger.info(
            "[SAGE_PARALLEL] Parent %s: %d local + %d remote = %d/%d",
            parent_id,
            local_done,
            remote_done,
            total_done,
            expected_total,
        )
        if total_done < expected_total:
            return

        # Drain ALL pending recompute events before launching the parent.
        # This ensures sage_process_layer has processed every queued
        # layer event (so layerwise_current_layer reflects the true
        # number of layers processed). Without this, the parent's blend
        # callback sees a stale current_layer and the layerwise strategy
        # starts injection from the wrong layer (race between drain and
        # launch).
        self._drain_recompute_events()

        # If early blend was triggered for this parent but the background
        # blend thread hasn't completed yet, defer the launch. The blend
        # thread will wake the busy loop when it finishes, and we'll
        # All done — build and launch parent.
        if parent_id not in self.parent_requests:
            self.parent_requests[parent_id] = (
                self._build_concurrent_parent_request(parent_id)
            )
        self._launch_final_request(parent_id)

    def _launch_final_request(self, parent_id: str) -> None:
        """Launch final request for generation after chunk prefilling.

        ZERO-COPY approach:
        1. Transfer chunk blocks directly to parent (no GPU copy needed!)
        2. Set num_computed_tokens so scheduler knows most tokens are done
        3. Parent uses transferred blocks directly for decode
        
        Chunks are always prefilled starting at position 0.
        RoPE adjustment is done in the adapter during blending.
        """
        parent_request = self.parent_requests[parent_id]
        prompt_len = len(parent_request.prompt_token_ids)
        
        # Compute chunk boundaries for RoPE adjustment during blending
        # Chunks are prefilled at position 0; boundaries tell the blender
        # which positions need RoPE correction.
        if parent_request.sage_chunk_boundaries is None:
            # Use position-ordered chunk IDs to compute boundaries correctly.
            # chunk_groups order is ingestion order which may differ from
            # position order for concurrent chunks.
            chunk_positions = self.concurrent_chunk_positions.get(parent_id, {})
            ordered_chunk_ids = [
                chunk_positions[pos]
                for pos in range(len(chunk_positions))
            ]
            chunk_boundaries = [0]
            position = 0
            for chunk_id in ordered_chunk_ids:
                chunk_len = self.chunk_lengths.get(chunk_id, 0)
                position += chunk_len
                chunk_boundaries.append(position)
            # Remove the last boundary (it's the total length, not a chunk start)
            chunk_boundaries = chunk_boundaries[:-1]
            parent_request.sage_chunk_boundaries = chunk_boundaries
            logger.info(
                f"[SAGE_BLEND] Computed chunk_boundaries from chunk lengths: {chunk_boundaries}"
            )
        
        import time as _time
        _t_launch = _time.perf_counter()
        logger.info("[SAGE_TIMING] launch_start t=%.6f", _t_launch)

        # Always do zero-copy transfer when SAGE is enabled
        if self.enable_sage:
            transfer_result = self.scheduler.transfer_sage_blocks_to_parent_zero_copy(parent_id)
        else:
            transfer_result = None
            
        if transfer_result:
            total_tokens, total_blocks, ordered_blocks_by_manager = transfer_result

            if total_tokens != prompt_len:
                raise RuntimeError(
                    "SAGE zero-copy transferred token count does not match parent "
                    f"prompt length for request {parent_id}: "
                    f"transferred={total_tokens} prompt_len={prompt_len}"
                )

            # The vLLM 1-token suffix prefill at position prompt_len-1
            # is needed to produce logits for the first output token.
            # However, it uses paged FlashAttention which is not batch-
            # invariant in FA2, causing hash mismatches between solo
            # and batched runs. At ptt<1.0 the suffix rerecompute
            # fused inject overwrites the batch-sensitive values.
            parent_request.num_computed_tokens = min(
                total_tokens, prompt_len - 1,
            )

            logger.info(
                f"[SAGE_ZERO_COPY] Parent {parent_id}: prompt_len={prompt_len}, "
                f"transferred {total_tokens} tokens, {total_blocks} blocks, "
                f"num_computed_tokens={parent_request.num_computed_tokens}"
            )

            # Mark that this request has pre-assigned blocks (for LMCache to skip loading)
            parent_request.sage_blocks_transferred = True
            logger.info(
                f"[SAGE_ZERO_COPY] Set sage_blocks_transferred=True for {parent_id}"
            )
        else:
            logger.info(
                f"Parent request {parent_id}: prompt_len={prompt_len}, "
                f"using CPU path (will retrieve from LMCache)"
            )
            parent_request.sage_blocks_transferred = False
            # Free chunk blocks since we're using CPU path - KV will be retrieved from LMCache
            self.scheduler.free_sage_chunk_blocks(parent_id)
        # Hash KV cache for this parent RIGHT BEFORE decode starts.
        # Add parent request to scheduler
        # Since blocks are already in req_to_blocks, allocate_slots will only
        # allocate blocks for the remaining (1 or few) tokens
        logger.info(
            f"[SAGE_ZERO_COPY] Adding parent {parent_id} to scheduler, "
            f"sage_blocks_transferred={parent_request.sage_blocks_transferred}"
        )

        self._partial_cleanup_chunk_state(parent_id)
        if self._sage_parent_fits(parent_request):
            self.scheduler.add_request(parent_request)
        else:
            self._sage_pending_parents.append((parent_id, parent_request))
            logger.info(
                "[SAGE_BACKPRESSURE] Queued parent %s; pending=%d "
                "(KV too tight to admit immediately)",
                parent_id, len(self._sage_pending_parents),
            )

        logger.info(
            "[SAGE_TIMING] launch_done t=%.6f elapsed=%.3fms",
            _time.perf_counter(),
            (_time.perf_counter() - _t_launch) * 1000,
        )
        logger.info(f"Final request {parent_id} first 100 token ids: {parent_request.prompt_token_ids[:100]}")

    def _sage_check_deadlock(self) -> None:
        """Detect SAGE chunk-pin deadlock and fail fast.

        Symptom: KV is near-full, our SAGE pre-WAITING queues are
        non-empty, and the queue size hasn't changed for many steps —
        meaning chunks of one parent finished prefill and are pinned in
        SAGE preserve, sibling chunks can't admit because KV is full,
        and no one can release blocks (preserved chunks aren't
        preemptible by vLLM). Without intervention this loops forever
        in scheduler retries, eventually the API server times out, and
        all clients see 500.

        We fail fast instead — raise RuntimeError after sustained no
        progress under pressure, so the cause is visible in logs and
        the user gets an actionable message.
        """
        try:
            block_pool = self.scheduler.kv_cache_manager.block_pool
        except AttributeError:
            return  # non-paged backend: no KV pool to check

        free = block_pool.get_num_free_blocks()
        total = block_pool.num_gpu_blocks
        util = 1.0 - (free / total) if total > 0 else 0.0

        queue_size = (
            len(self._sage_pending_chunks)
            + len(self._sage_pending_parents)
        )

        # No progress = queue size unchanged AND non-empty.
        no_progress = (
            queue_size > 0 and queue_size == self._sage_last_queue_size
        )
        self._sage_last_queue_size = queue_size

        if util > 0.95 and no_progress:
            self._sage_stuck_steps += 1
        else:
            self._sage_stuck_steps = 0

        # ~200 consecutive stuck steps ≈ a few seconds of true stall.
        # Tune up if false positives appear at heavy-but-progressing load.
        if self._sage_stuck_steps > 200:
            stalled_parents = [
                pid
                for pid, expected in self.concurrent_total_chunks.items()
                if 0 < len(self.chunk_completion.get(pid, set())) < expected
            ]
            raise RuntimeError(
                f"SAGE concurrent-prefill deadlock detected: "
                f"KV at {util*100:.1f}% utilization, "
                f"{queue_size} requests pending in SAGE pre-WAITING, "
                f"{len(stalled_parents)} parent(s) partial-pin "
                f"(chunks finished but cannot blend due to KV exhaustion). "
                f"This happens when concurrent SAGE chunk admissions "
                f"exceed the KV cache capacity. Reduce --concurrency, "
                f"shorten prompts, or increase --gpu-memory-utilization."
            )

    def _sage_request_block_need(self, request, kvm) -> int:
        """Non-destructive estimate of how many new blocks `request`
        would need from `kvm.allocate_slots` right now.
        """
        prompt_len = len(request.prompt_token_ids)
        num_tokens_need_slot = min(prompt_len, kvm.max_model_len)
        return kvm.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=kvm.empty_kv_cache_blocks.blocks,
            num_encoder_tokens=0,
        )

    def _sage_pending_waiting_demand(self, kvm) -> int:
        """Sum the block demand of requests already in vLLM's WAITING
        queue. They were admitted before us and will be allocated by
        scheduler.schedule() ahead of any newly-admitted request, so
        their demand must be subtracted from free blocks before we
        decide whether the new request fits. This is what closes the
        point-in-time hole — without it, every chunk in a 48-burst
        sees `free_blocks` as if no other chunk had been admitted.
        Cost: O(len(waiting)), small at our scale.
        """
        total = 0
        for req in self.scheduler.waiting:
            total += self._sage_request_block_need(req, kvm)
        return total

    def _sage_parent_fits(self, parent_request) -> bool:
        """Whether `parent_request` can be admitted into the scheduler now.

        Mirrors the two gates in vLLM's WAITING-admit loop
        (scheduler.schedule()):

          1. `len(running) >= max_num_running_reqs` — refuse.
          2. `allocate_slots(...) is not None`, which boils down to
             `num_blocks_to_allocate <= num_free_blocks`.

        Gate (2) is replicated non-destructively by calling the same
        `coordinator.get_num_blocks_to_allocate` that allocate_slots
        uses internally. Because admit happens BEFORE any allocation,
        we also subtract the demand of requests already in WAITING
        (admitted before us this same step) so a burst of admits is
        accounted cumulatively, not point-in-time.
        """
        sched = self.scheduler

        if len(sched.running) >= sched.max_num_running_reqs:
            return False

        try:
            kvm = sched.kv_cache_manager
        except AttributeError:
            return True

        needed = self._sage_request_block_need(parent_request, kvm)
        free = kvm.block_pool.get_num_free_blocks()
        pending = self._sage_pending_waiting_demand(kvm)
        return needed <= free - pending

    def _sage_chunk_fits(self, chunk_request) -> bool:
        """Whether `chunk_request` can be admitted into the scheduler now.

        Same predicate as `_sage_parent_fits`. The only difference is
        the magnitude: a fresh chunk has 0 existing blocks for its
        request_id, so the coordinator returns the full prompt's
        worth (~ceil(prompt_len / block_size)) — much more than the
        ~0-1 a zero-copied parent needs.
        """
        sched = self.scheduler

        if len(sched.running) >= sched.max_num_running_reqs:
            return False

        try:
            kvm = sched.kv_cache_manager
        except AttributeError:
            return True

        needed = self._sage_request_block_need(chunk_request, kvm)
        free = kvm.block_pool.get_num_free_blocks()
        pending = self._sage_pending_waiting_demand(kvm)
        return needed <= free - pending

    def _drain_sage_pending_chunks(self) -> None:
        """Admit queued SAGE chunks while KV has room. FIFO by submit
        time so chunks for one parent generally admit close together.
        """
        admitted = 0
        while self._sage_pending_chunks:
            request = self._sage_pending_chunks[0]
            if not self._sage_chunk_fits(request):
                break
            self._sage_pending_chunks.popleft()
            self.scheduler.add_request(request)
            admitted += 1
            logger.info(
                "[SAGE_BACKPRESSURE] Admitted chunk %s from queue; "
                "remaining_chunks=%d",
                request.request_id, len(self._sage_pending_chunks),
            )
        if admitted == 0 and self._sage_pending_chunks:
            logger.debug(
                "[SAGE_BACKPRESSURE] %d chunk(s) waiting for KV room",
                len(self._sage_pending_chunks),
            )

    def _drain_sage_pending_parents(self) -> None:
        """Admit queued SAGE parents while KV has room.

        Called at the top of step() — runs after the previous step's
        finished requests have freed their blocks, so newly-available
        space immediately admits backlogged parents in FIFO order.
        """
        admitted = 0
        while self._sage_pending_parents:
            parent_id, parent_request = self._sage_pending_parents[0]
            if not self._sage_parent_fits(parent_request):
                break
            self._sage_pending_parents.popleft()
            self.scheduler.add_request(parent_request)
            admitted += 1
            logger.info(
                "[SAGE_BACKPRESSURE] Admitted parent %s from queue; "
                "remaining=%d",
                parent_id, len(self._sage_pending_parents),
            )
        if admitted == 0 and self._sage_pending_parents:
            # Optional: surface tightness once per step (debug-level so
            # we don't spam at INFO).
            logger.debug(
                "[SAGE_BACKPRESSURE] %d parent(s) waiting for KV room",
                len(self._sage_pending_parents),
            )

    def _partial_cleanup_chunk_state(self, parent_id: str) -> None:
        """Partial cleanup - preserves chunk blocks for blending."""
        chunk_ids = self.chunk_groups.get(parent_id, [])
        for chunk_id in chunk_ids:
            self.chunk_lengths.pop(chunk_id, None)
            self._chunk_token_ids_by_req_id.pop(chunk_id, None)
            self._chunk_req_to_position.pop(chunk_id, None)
        self.concurrent_parent_template.pop(parent_id, None)
        self.concurrent_parent_sampling_params.pop(parent_id, None)
        self.concurrent_chunk_payloads.pop(parent_id, None)
        self.concurrent_chunk_positions.pop(parent_id, None)
        self.concurrent_total_chunks.pop(parent_id, None)
        self._concurrent_query_token_count.pop(parent_id, None)
        self._concurrent_chunk_mm_features.pop(parent_id, None)
        # SAGE parallel prefill state
        self._sage_remote_chunks_received.pop(parent_id, None)
        self._sage_remote_drains_done.pop(parent_id, None)
        self._sage_local_positions.pop(parent_id, None)
        self._sage_recv_started.pop(parent_id, None)
        self._sage_remote_image_grid_thw.pop(parent_id, None)
        with self._sage_layer_arrivals_lock:
            for k in list(self._sage_layer_arrivals):
                if k[0] == parent_id:
                    self._sage_layer_arrivals.pop(k, None)
        # Clean up parent tracking but NOT the chunk_to_parent mapping
        # as we need that to track which chunks belong to which parent
        self.chunk_groups.pop(parent_id, None)
        self.chunk_completion.pop(parent_id, None)
        self.parent_requests.pop(parent_id, None)

    def _cleanup_chunk_state(self, parent_id: str) -> None:
        """Clean up state tracking for completed concurrent prefill."""
        chunk_ids = self.chunk_groups.get(parent_id, [])

        # Clean up per-chunk tracking
        for chunk_id in chunk_ids:
            self.chunk_to_parent.pop(chunk_id, None)
            self.chunk_lengths.pop(chunk_id, None)
            self._chunk_token_ids_by_req_id.pop(chunk_id, None)
            self._chunk_req_to_position.pop(chunk_id, None)

        # Clean up per-parent tracking
        self.chunk_groups.pop(parent_id, None)
        self.chunk_completion.pop(parent_id, None)
        self.parent_requests.pop(parent_id, None)
        self.concurrent_parent_template.pop(parent_id, None)
        self.concurrent_parent_sampling_params.pop(parent_id, None)
        self.concurrent_chunk_payloads.pop(parent_id, None)
        self.concurrent_chunk_positions.pop(parent_id, None)
        self.concurrent_total_chunks.pop(parent_id, None)
        self._concurrent_query_token_count.pop(parent_id, None)
        self._concurrent_chunk_mm_features.pop(parent_id, None)
        # SAGE parallel prefill state
        self._sage_remote_chunks_received.pop(parent_id, None)
        self._sage_remote_drains_done.pop(parent_id, None)
        self._sage_local_positions.pop(parent_id, None)
        self._sage_remote_image_grid_thw.pop(parent_id, None)
        self._sage_recv_started.pop(parent_id, None)
        with self._sage_layer_arrivals_lock:
            for k in list(self._sage_layer_arrivals):
                if k[0] == parent_id:
                    self._sage_layer_arrivals.pop(k, None)

    # ==================== End Concurrent Prefill Methods ====================

    def preprocess_add_request(self, request: EngineCoreRequest) -> tuple[Request, int]:
        """Preprocess the request.

        This function could be directly used in input processing thread to allow
        request initialization running in parallel with Model forward
        """
        # Note on thread safety: no race condition.
        # `mm_receiver_cache` is reset at the end of LLMEngine init,
        # and will only be accessed in the input processing thread afterwards.
        if self.mm_receiver_cache is not None and request.mm_features:
            request.mm_features = self.mm_receiver_cache.get_and_update_features(
                request.mm_features
            )

        req = Request.from_engine_core_request(request, self.request_block_hasher)
        if req.use_structured_output:
            # Note on thread safety: no race condition.
            # `grammar_init` is only invoked in input processing thread. For
            # `structured_output_manager`, each request is independent and
            # grammar compilation is async. Scheduler always checks grammar
            # compilation status before scheduling request.
            self.structured_output_manager.grammar_init(req)
        return req, request.current_wave


class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""

    ENGINE_CORE_DEAD = b"ENGINE_CORE_DEAD"

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
        engine_index: int = 0,
    ):
        self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
        self.output_queue = queue.Queue[tuple[int, EngineCoreOutputs] | bytes]()
        executor_fail_callback = lambda: self.input_queue.put_nowait(
            (EngineCoreRequestType.EXECUTOR_FAILED, b"")
        )

        self.engine_index = engine_index
        identity = self.engine_index.to_bytes(length=2, byteorder="little")
        self.engines_running = False

        with self._perform_handshakes(
            handshake_address,
            identity,
            local_client,
            vllm_config,
            client_handshake_address,
        ) as addresses:
            self.client_count = len(addresses.outputs)

            # Set up data parallel environment.
            self.has_coordinator = addresses.coordinator_output is not None
            self.frontend_stats_publish_address = (
                addresses.frontend_stats_publish_address
            )
            logger.debug(
                "Has DP Coordinator: %s, stats publish address: %s",
                self.has_coordinator,
                self.frontend_stats_publish_address,
            )
            # Only publish request queue stats to coordinator for "internal"
            # and "hybrid" LB modes .
            self.publish_dp_lb_stats = (
                self.has_coordinator
                and not vllm_config.parallel_config.data_parallel_external_lb
            )

            self._init_data_parallel(vllm_config)

            super().__init__(
                vllm_config, executor_class, log_stats, executor_fail_callback
            )

            # Background Threads and Queues for IO. These enable us to
            # overlap ZMQ socket IO with GPU since they release the GIL,
            # and to overlap some serialization/deserialization with the
            # model forward pass.
            # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
            ready_event = threading.Event()
            input_thread = threading.Thread(
                target=self.process_input_sockets,
                args=(
                    addresses.inputs,
                    addresses.coordinator_input,
                    identity,
                    ready_event,
                ),
                daemon=True,
            )
            input_thread.start()

            self.output_thread = threading.Thread(
                target=self.process_output_sockets,
                args=(
                    addresses.outputs,
                    addresses.coordinator_output,
                    self.engine_index,
                ),
                daemon=True,
            )
            self.output_thread.start()

            # Don't complete handshake until DP coordinator ready message is
            # received.
            while not ready_event.wait(timeout=10):
                if not input_thread.is_alive():
                    raise RuntimeError("Input socket thread died during startup")
                assert addresses.coordinator_input is not None
                logger.info("Waiting for READY message from DP Coordinator...")

    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        vllm_config: VllmConfig,
        client_handshake_address: str | None,
    ) -> Generator[EngineZmqAddresses, None, None]:
        """
        Perform startup handshakes.

        For DP=1 or offline mode, this is with the colocated front-end process.

        For DP>1 with internal load-balancing this is with the shared front-end
        process which may reside on a different node.

        For DP>1 with external or hybrid load-balancing, two handshakes are
        performed:
            - With the rank 0 front-end process which retrieves the
              DP Coordinator ZMQ addresses and DP process group address.
            - With the colocated front-end process which retrieves the
              client input/output socket addresses.
        with the exception of the rank 0 and colocated engines themselves which
        don't require the second handshake.

        Here, "front-end" process can mean the process containing the engine
        core client (which is the API server process in the case the API
        server is not scaled out), OR the launcher process running the
        run_multi_api_server() function in serve.py.
        """
        input_ctx = zmq.Context()
        is_local = local_client and client_handshake_address is None
        headless = not local_client
        handshake = self._perform_handshake(
            input_ctx,
            handshake_address,
            identity,
            is_local,
            headless,
            vllm_config,
            vllm_config.parallel_config,
        )
        if client_handshake_address is None:
            with handshake as addresses:
                yield addresses
        else:
            assert local_client
            local_handshake = self._perform_handshake(
                input_ctx, client_handshake_address, identity, True, False, vllm_config
            )
            with handshake as addresses, local_handshake as client_addresses:
                addresses.inputs = client_addresses.inputs
                addresses.outputs = client_addresses.outputs
                yield addresses

        # Update config which may have changed from the handshake
        vllm_config.__post_init__()

    @contextmanager
    def _perform_handshake(
        self,
        ctx: zmq.Context,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        headless: bool,
        vllm_config: VllmConfig,
        parallel_config_to_update: ParallelConfig | None = None,
    ) -> Generator[EngineZmqAddresses, None, None]:
        with make_zmq_socket(
            ctx,
            handshake_address,
            zmq.DEALER,
            identity=identity,
            linger=5000,
            bind=False,
        ) as handshake_socket:
            # Register engine with front-end.
            addresses = self.startup_handshake(
                handshake_socket, local_client, headless, parallel_config_to_update
            )
            yield addresses

            # Send ready message.
            num_gpu_blocks = vllm_config.cache_config.num_gpu_blocks
            # We pass back the coordinator stats update address here for the
            # external LB case for our colocated front-end to use (coordinator
            # only runs with rank 0).
            dp_stats_address = self.frontend_stats_publish_address

            # Include config hash for DP configuration validation
            ready_msg = {
                "status": "READY",
                "local": local_client,
                "headless": headless,
                "num_gpu_blocks": num_gpu_blocks,
                "dp_stats_address": dp_stats_address,
            }
            if vllm_config.parallel_config.data_parallel_size > 1:
                ready_msg["parallel_config_hash"] = (
                    vllm_config.parallel_config.compute_hash()
                )

            handshake_socket.send(msgspec.msgpack.encode(ready_msg))

    @staticmethod
    def startup_handshake(
        handshake_socket: zmq.Socket,
        local_client: bool,
        headless: bool,
        parallel_config: ParallelConfig | None = None,
    ) -> EngineZmqAddresses:
        # Send registration message.
        handshake_socket.send(
            msgspec.msgpack.encode(
                {
                    "status": "HELLO",
                    "local": local_client,
                    "headless": headless,
                }
            )
        )

        # Receive initialization message.
        logger.debug("Waiting for init message from front-end.")
        if not handshake_socket.poll(timeout=HANDSHAKE_TIMEOUT_MINS * 60_000):
            raise RuntimeError(
                "Did not receive response from front-end "
                f"process within {HANDSHAKE_TIMEOUT_MINS} "
                f"minutes"
            )
        init_bytes = handshake_socket.recv()
        init_message: EngineHandshakeMetadata = msgspec.msgpack.decode(
            init_bytes, type=EngineHandshakeMetadata
        )
        logger.debug("Received init message: %s", init_message)

        if parallel_config is not None:
            for key, value in init_message.parallel_config.items():
                setattr(parallel_config, key, value)

        return init_message.addresses

    @staticmethod
    def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
        """Launch EngineCore busy loop in background process."""

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        # Ensure we can serialize transformer config after spawning
        maybe_register_config_serialize_by_value()

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the engine_core
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        engine_core: EngineCoreProc | None = None
        try:
            parallel_config: ParallelConfig = kwargs["vllm_config"].parallel_config
            if parallel_config.data_parallel_size > 1 or dp_rank > 0:
                set_process_title("EngineCore", f"DP{dp_rank}")
                decorate_logs()
                # Set data parallel rank for this engine process.
                parallel_config.data_parallel_rank = dp_rank
                parallel_config.data_parallel_rank_local = local_dp_rank
                engine_core = DPEngineCoreProc(*args, **kwargs)
            else:
                set_process_title("EngineCore")
                decorate_logs()
                engine_core = EngineCoreProc(*args, **kwargs)

            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception as e:
            if engine_core is None:
                logger.exception("EngineCore failed to start.")
            else:
                logger.exception("EngineCore encountered a fatal error.")
                engine_core._send_engine_dead()
            raise e
        finally:
            if engine_core is not None:
                engine_core.shutdown()

    def _init_data_parallel(self, vllm_config: VllmConfig):
        pass

    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""

        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            # 2) Step the engine core and return the outputs.
            self._process_engine_step()

    def _process_input_queue(self):
        """Exits when an engine step needs to be performed."""

        waited = False
        while (
            not self.engines_running
            and not self.scheduler.has_requests()
            and not self.batch_queue
        ):
            if self.input_queue.empty():
                with self.aborts_queue.mutex:
                    self.aborts_queue.queue.clear()
                if logger.isEnabledFor(DEBUG):
                    logger.debug("EngineCore waiting for work.")
                    waited = True
            req = self.input_queue.get()
            # SAGE sentinels from P2pNcclEngine callbacks.
            if req == ("_SAGE_REMOTE_KV",):
                self._process_remote_chunk_arrivals()
                continue
            if isinstance(req, tuple) and req and req[0] == "_SAGE_RECOMPUTE_LAYER":
                self._sage_run_pipelined_recompute(req[1], req[2])
                continue
            self._handle_client_request(*req)

        if waited:
            logger.debug("EngineCore loop active.")

        # Handle any more client requests.
        while not self.input_queue.empty():
            req = self.input_queue.get_nowait()
            if req == ("_SAGE_REMOTE_KV",):
                self._process_remote_chunk_arrivals()
                continue
            if isinstance(req, tuple) and req and req[0] == "_SAGE_RECOMPUTE_LAYER":
                self._sage_run_pipelined_recompute(req[1], req[2])
                continue
            self._handle_client_request(*req)

    def _process_engine_step(self) -> bool:
        """Called only when there are unfinished local requests."""

        # Step the engine core.
        outputs, model_executed = self.step_fn()
        # Put EngineCoreOutputs into the output queue.
        for output in outputs.items() if outputs else ():
            self.output_queue.put_nowait(output)
        # Post-step hook.
        self.post_step(model_executed)

        return model_executed

    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        """Dispatch request from client."""

        if request_type == EngineCoreRequestType.ADD:
            req, request_wave = request
            self.add_request(req, request_wave)
        elif request_type == EngineCoreRequestType.ABORT:
            self.abort_requests(request)
        elif request_type == EngineCoreRequestType.UTILITY:
            client_idx, call_id, method_name, args = request
            output = UtilityOutput(call_id)
            try:
                method = getattr(self, method_name)
                result = method(*self._convert_msgspec_args(method, args))
                output.result = UtilityResult(result)
            except BaseException as e:
                logger.exception("Invocation of %s method failed", method_name)
                output.failure_message = (
                    f"Call to {method_name} method failed: {str(e)}"
                )
            self.output_queue.put_nowait(
                (client_idx, EngineCoreOutputs(utility_output=output))
            )
        elif request_type == EngineCoreRequestType.EXECUTOR_FAILED:
            raise RuntimeError("Executor failed.")
        else:
            logger.error(
                "Unrecognized input request type encountered: %s", request_type
            )

    @staticmethod
    def _convert_msgspec_args(method, args):
        """If a provided arg type doesn't match corresponding target method
        arg type, try converting to msgspec object."""
        if not args:
            return args
        arg_types = signature(method).parameters.values()
        assert len(args) <= len(arg_types)
        return tuple(
            msgspec.convert(v, type=p.annotation)
            if isclass(p.annotation)
            and issubclass(p.annotation, msgspec.Struct)
            and not isinstance(v, p.annotation)
            else v
            for v, p in zip(args, arg_types)
        )

    def _send_engine_dead(self):
        """Send EngineDead status to the EngineCoreClient."""

        # Put ENGINE_CORE_DEAD in the queue.
        self.output_queue.put_nowait(EngineCoreProc.ENGINE_CORE_DEAD)

        # Wait until msg sent by the daemon before shutdown.
        self.output_thread.join(timeout=5.0)
        if self.output_thread.is_alive():
            logger.fatal(
                "vLLM shutdown signal from EngineCore failed "
                "to send. Please report this issue."
            )

    def process_input_sockets(
        self,
        input_addresses: list[str],
        coord_input_address: str | None,
        identity: bytes,
        ready_event: threading.Event,
    ):
        """Input socket IO thread."""

        # Msgpack serialization decoding.
        add_request_decoder = MsgpackDecoder(EngineCoreRequest)
        generic_decoder = MsgpackDecoder()

        with ExitStack() as stack, zmq.Context() as ctx:
            input_sockets = [
                stack.enter_context(
                    make_zmq_socket(
                        ctx, input_address, zmq.DEALER, identity=identity, bind=False
                    )
                )
                for input_address in input_addresses
            ]
            if coord_input_address is None:
                coord_socket = None
            else:
                coord_socket = stack.enter_context(
                    make_zmq_socket(
                        ctx,
                        coord_input_address,
                        zmq.XSUB,
                        identity=identity,
                        bind=False,
                    )
                )
                # Send subscription message to coordinator.
                coord_socket.send(b"\x01")

            # Register sockets with poller.
            poller = zmq.Poller()
            for input_socket in input_sockets:
                # Send initial message to each input socket - this is required
                # before the front-end ROUTER socket can send input messages
                # back to us.
                input_socket.send(b"")
                poller.register(input_socket, zmq.POLLIN)

            if coord_socket is not None:
                # Wait for ready message from coordinator.
                assert coord_socket.recv() == b"READY"
                poller.register(coord_socket, zmq.POLLIN)

            ready_event.set()
            del ready_event
            while True:
                for input_socket, _ in poller.poll():
                    # (RequestType, RequestData)
                    type_frame, *data_frames = input_socket.recv_multipart(copy=False)
                    request_type = EngineCoreRequestType(bytes(type_frame.buffer))

                    # Deserialize the request data.
                    if request_type == EngineCoreRequestType.ADD:
                        request = add_request_decoder.decode(data_frames)
                        request = self.preprocess_add_request(request)
                    else:
                        request = generic_decoder.decode(data_frames)

                        if request_type == EngineCoreRequestType.ABORT:
                            # Aborts are added to *both* queues, allows us to eagerly
                            # process aborts while also ensuring ordering in the input
                            # queue to avoid leaking requests. This is ok because
                            # aborting in the scheduler is idempotent.
                            self.aborts_queue.put_nowait(request)

                    # Push to input queue for core busy loop.
                    self.input_queue.put_nowait((request_type, request))

    def process_output_sockets(
        self,
        output_paths: list[str],
        coord_output_path: str | None,
        engine_index: int,
    ):
        """Output socket IO thread."""

        # Msgpack serialization encoding.
        encoder = MsgpackEncoder()
        # Send buffers to reuse.
        reuse_buffers: list[bytearray] = []
        # Keep references to outputs and buffers until zmq is finished
        # with them (outputs may contain tensors/np arrays whose
        # backing buffers were extracted for zero-copy send).
        pending = deque[tuple[zmq.MessageTracker, Any, bytearray]]()

        # We must set linger to ensure the ENGINE_CORE_DEAD
        # message is sent prior to closing the socket.
        with ExitStack() as stack, zmq.Context() as ctx:
            sockets = [
                stack.enter_context(
                    make_zmq_socket(ctx, output_path, zmq.PUSH, linger=4000)
                )
                for output_path in output_paths
            ]
            coord_socket = (
                stack.enter_context(
                    make_zmq_socket(
                        ctx, coord_output_path, zmq.PUSH, bind=False, linger=4000
                    )
                )
                if coord_output_path is not None
                else None
            )
            max_reuse_bufs = len(sockets) + 1

            while True:
                output = self.output_queue.get()
                if output == EngineCoreProc.ENGINE_CORE_DEAD:
                    for socket in sockets:
                        socket.send(output)
                    break
                assert not isinstance(output, bytes)
                client_index, outputs = output
                outputs.engine_index = engine_index

                if client_index == -1:
                    # Don't reuse buffer for coordinator message
                    # which will be very small.
                    assert coord_socket is not None
                    coord_socket.send_multipart(encoder.encode(outputs))
                    continue

                # Reclaim buffers that zmq is finished with.
                while pending and pending[-1][0].done:
                    reuse_buffers.append(pending.pop()[2])

                buffer = reuse_buffers.pop() if reuse_buffers else bytearray()
                buffers = encoder.encode_into(outputs, buffer)
                tracker = sockets[client_index].send_multipart(
                    buffers, copy=False, track=True
                )
                if not tracker.done:
                    ref = outputs if len(buffers) > 1 else None
                    pending.appendleft((tracker, ref, buffer))
                elif len(reuse_buffers) < max_reuse_bufs:
                    # Limit the number of buffers to reuse.
                    reuse_buffers.append(buffer)


class DPEngineCoreProc(EngineCoreProc):
    """ZMQ-wrapper for running EngineCore in background process
    in a data parallel context."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
    ):
        # Counts forward-passes of the model so that we can synchronize
        # finished with DP peers every N steps.
        self.step_counter = 0
        self.current_wave = 0
        self.last_counts = (0, 0)

        # Initialize the engine.
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        super().__init__(
            vllm_config,
            local_client,
            handshake_address,
            executor_class,
            log_stats,
            client_handshake_address,
            dp_rank,
        )

    def _init_data_parallel(self, vllm_config: VllmConfig):
        # Configure GPUs and stateless process group for data parallel.
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        dp_size = vllm_config.parallel_config.data_parallel_size
        local_dp_rank = vllm_config.parallel_config.data_parallel_rank_local

        assert dp_size > 1
        assert local_dp_rank is not None
        assert 0 <= local_dp_rank <= dp_rank < dp_size

        if vllm_config.kv_transfer_config is not None:
            # modify the engine_id and append the local_dp_rank to it to ensure
            # that the kv_transfer_config is unique for each DP rank.
            vllm_config.kv_transfer_config.engine_id = (
                f"{vllm_config.kv_transfer_config.engine_id}_dp{local_dp_rank}"
            )
            logger.debug(
                "Setting kv_transfer_config.engine_id to %s",
                vllm_config.kv_transfer_config.engine_id,
            )

        self.dp_rank = dp_rank
        self.dp_group = vllm_config.parallel_config.stateless_init_dp_group()

    def shutdown(self):
        super().shutdown()
        if dp_group := getattr(self, "dp_group", None):
            stateless_destroy_torch_distributed_process_group(dp_group)

    def add_request(self, request: Request, request_wave: int = 0):
        if self.has_coordinator and request_wave != self.current_wave:
            if request_wave > self.current_wave:
                self.current_wave = request_wave
            elif not self.engines_running:
                # Request received for an already-completed wave, notify
                # front-end that we need to start the next one.
                self.output_queue.put_nowait(
                    (-1, EngineCoreOutputs(start_wave=self.current_wave))
                )

        super().add_request(request, request_wave)

    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        if request_type == EngineCoreRequestType.START_DP_WAVE:
            new_wave, exclude_eng_index = request
            if exclude_eng_index != self.engine_index and (
                new_wave >= self.current_wave
            ):
                self.current_wave = new_wave
                if not self.engines_running:
                    logger.debug("EngineCore starting idle loop for wave %d.", new_wave)
                    self.engines_running = True
        else:
            super()._handle_client_request(request_type, request)

    def _maybe_publish_request_counts(self):
        if not self.publish_dp_lb_stats:
            return

        # Publish our request counts (if they've changed).
        counts = self.scheduler.get_request_counts()
        if counts != self.last_counts:
            self.last_counts = counts
            stats = SchedulerStats(
                *counts, step_counter=self.step_counter, current_wave=self.current_wave
            )
            self.output_queue.put_nowait((-1, EngineCoreOutputs(scheduler_stats=stats)))

    def run_busy_loop(self):
        """Core busy loop of the EngineCore for data parallel case."""

        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()

            # 2) Step the engine core.
            executed = self._process_engine_step()
            self._maybe_publish_request_counts()

            local_unfinished_reqs = self.scheduler.has_unfinished_requests()
            if not executed:
                if not local_unfinished_reqs and not self.engines_running:
                    # All engines are idle.
                    continue

                # We are in a running state and so must execute a dummy pass
                # if the model didn't execute any ready requests.
                self.execute_dummy_batch()

            # 3) All-reduce operation to determine global unfinished reqs.
            self.engines_running = self._has_global_unfinished_reqs(
                local_unfinished_reqs
            )

            if not self.engines_running:
                if self.dp_rank == 0 or not self.has_coordinator:
                    # Notify client that we are pausing the loop.
                    logger.debug(
                        "Wave %d finished, pausing engine loop.", self.current_wave
                    )
                    # In the coordinator case, dp rank 0 sends updates to the
                    # coordinator. Otherwise (offline spmd case), each rank
                    # sends the update to its colocated front-end process.
                    client_index = -1 if self.has_coordinator else 0
                    self.output_queue.put_nowait(
                        (
                            client_index,
                            EngineCoreOutputs(wave_complete=self.current_wave),
                        )
                    )
                # Increment wave count and reset step counter.
                self.current_wave += 1
                self.step_counter = 0

    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
        # Optimization - only perform finish-sync all-reduce every 32 steps.
        self.step_counter += 1
        if self.step_counter % 32 != 0:
            return True

        return ParallelConfig.has_unfinished_dp(self.dp_group, local_unfinished)

    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        stateless_destroy_torch_distributed_process_group(self.dp_group)
        self.shutdown()

        parallel_config = self.vllm_config.parallel_config
        old_dp_size = parallel_config.data_parallel_size
        parallel_config.data_parallel_size = reconfig_request.new_data_parallel_size
        if reconfig_request.new_data_parallel_rank != -1:
            parallel_config.data_parallel_rank = reconfig_request.new_data_parallel_rank
        # local rank specifies device visibility, it should not be changed
        assert (
            reconfig_request.new_data_parallel_rank_local
            == ReconfigureRankType.KEEP_CURRENT_RANK
        )
        parallel_config.data_parallel_master_ip = (
            reconfig_request.new_data_parallel_master_ip
        )
        parallel_config.data_parallel_master_port = (
            reconfig_request.new_data_parallel_master_port
        )
        if reconfig_request.new_data_parallel_rank != -2:
            self.dp_rank = parallel_config.data_parallel_rank
            self.dp_group = parallel_config.stateless_init_dp_group()
        reconfig_request.new_data_parallel_master_port = (
            parallel_config.data_parallel_master_port
        )

        self.model_executor.reinitialize_distributed(reconfig_request)
        if reconfig_request.new_data_parallel_size > old_dp_size:
            assert self.available_gpu_memory_for_kv_cache > 0
            # pass available_gpu_memory_for_kv_cache from existing
            # engine-cores to new engine-cores so they can directly
            # use it in _initialize_kv_caches() rather than profiling.
            ParallelConfig.sync_kv_cache_memory_size(
                self.dp_group, self.available_gpu_memory_for_kv_cache
            )
            # NOTE(yongji): newly joined workers require dummy_run even
            # CUDA graph is not used
            self.model_executor.collective_rpc("compile_or_warm_up_model")
        if (
            reconfig_request.new_data_parallel_rank
            == ReconfigureRankType.SHUTDOWN_CURRENT_RANK
        ):
            self.shutdown()
            logger.info("DPEngineCoreProc %s shutdown", self.dp_rank)
        else:
            logger.info(
                "Distributed environment reinitialized for DP rank %s", self.dp_rank
            )


class DPEngineCoreActor(DPEngineCoreProc):
    """
    Ray actor for running EngineCore in a data parallel context
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        self.addresses = addresses
        vllm_config.parallel_config.data_parallel_rank = dp_rank
        vllm_config.parallel_config.data_parallel_rank_local = local_dp_rank

        # Set CUDA_VISIBLE_DEVICES as early as possible in actor life cycle
        # NOTE: in MP we set CUDA_VISIBLE_DEVICES at process creation time,
        # and this cannot be done in the same way for Ray because:
        # 1) Ray manages life cycle of all ray workers (including
        # DPEngineCoreActor)
        # 2) Ray sets CUDA_VISIBLE_DEVICES based on num_gpus configuration
        # To bypass 2, we need to also set
        # RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES, but vLLM workers created
        # thereafter would have CUDA_VISIBLE_DEVICES set, which is sticky:
        # https://github.com/ray-project/ray/blob/e752fc319ddedd9779a0989b6d3613909bad75c9/python/ray/_private/worker.py#L456 # noqa: E501
        # This is problematic because when the vLLM worker (a Ray actor)
        # executes a task, it indexes into the sticky CUDA_VISIBLE_DEVICES
        # rather than directly using the GPU ID, potentially resulting in
        # index out of bounds error. See:
        # https://github.com/ray-project/ray/pull/40461/files#diff-31e8159767361e4bc259b6d9883d9c0d5e5db780fcea4a52ead4ee3ee4a59a78R1860 # noqa: E501
        # and get_accelerator_ids_for_accelerator_resource() in worker.py
        # of ray.
        self._set_visible_devices(vllm_config, local_dp_rank)

        super().__init__(vllm_config, local_client, "", executor_class, log_stats)

    def _set_visible_devices(self, vllm_config: VllmConfig, local_dp_rank: int):
        from vllm.platforms import current_platform

        if current_platform.is_xpu():
            pass
        else:
            device_control_env_var = current_platform.device_control_env_var
            self._set_cuda_visible_devices(
                vllm_config, local_dp_rank, device_control_env_var
            )

    def _set_cuda_visible_devices(
        self, vllm_config: VllmConfig, local_dp_rank: int, device_control_env_var: str
    ):
        world_size = vllm_config.parallel_config.world_size
        # Set CUDA_VISIBLE_DEVICES or equivalent.
        try:
            value = get_device_indices(
                device_control_env_var, local_dp_rank, world_size
            )
            os.environ[device_control_env_var] = value
        except IndexError as e:
            raise Exception(
                f"Error setting {device_control_env_var}: "
                f"local range: [{local_dp_rank * world_size}, "
                f"{(local_dp_rank + 1) * world_size}) "
                f'base value: "{os.getenv(device_control_env_var)}"'
            ) from e

    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        vllm_config: VllmConfig,
        client_handshake_address: str | None,
    ):
        """
        For Ray, we don't need to actually perform handshake.
        All addresses information is known before the actor creation.
        Therefore, we simply yield these addresses.
        """
        yield self.addresses

    def wait_for_init(self):
        """
        Wait until the engine core is initialized.

        This is just an empty method. When ray.get() on this method
        (or any other method of the actor) returns, it is guaranteed
        that actor creation (i.e., __init__) is complete.
        """
        pass

    def run(self):
        """
        Run the engine core busy loop.
        """
        try:
            self.run_busy_loop()
        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception:
            logger.exception("EngineCore encountered a fatal error.")
            raise
        finally:
            self.shutdown()
