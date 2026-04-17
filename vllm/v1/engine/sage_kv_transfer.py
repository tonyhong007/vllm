# SPDX-License-Identifier: Apache-2.0
"""SAGE Cross-GPU KV Block Transfer.

Transfers KV cache blocks between GPUs using NCCL via P2pNcclEngine.

Two modes:
- Bulk (default): All layers sent as one concatenated tensor after forward.
- Per-layer (SAGE_PER_LAYER_TRANSFER=1): Each layer sent during forward
  via P2pNcclConnector.save_kv_layer. Better for slow interconnect.
"""
import torch

from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine import (
    P2pNcclEngine,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


class SageKVTransferEngine:
    """Handles GPU-to-GPU KV cache block transfer using P2pNcclEngine.

    One instance per DP engine.  Uses P2pNcclEngine for NCCL comm setup
    (ZMQ signaling + ncclCommInitRank) and tensor send/recv.
    """

    def __init__(
        self,
        dp_rank: int,
        dp_size: int,
        local_gpu_id: int,
        kv_transfer_config: KVTransferConfig,
    ):
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.local_gpu_id = local_gpu_id
        self.device = torch.device(f"cuda:{local_gpu_id}")
        # KV cache tensors — set after model init.
        self._kv_caches: list[torch.Tensor] | None = None
        self._num_layers: int = 0
        self._is_flash_attn: bool = False

        # Create P2pNcclEngine with dp_rank as port_offset so each
        # DP engine gets a unique port: kv_port + dp_rank.
        self._nccl_engine = P2pNcclEngine(
            local_rank=local_gpu_id,
            config=kv_transfer_config,
            port_offset=dp_rank,
        )
        # Callback set by EngineCore to wake it when remote KV arrives.
        self._nccl_engine._on_recv_callback = None

        logger.info(
            "[SAGE_KV_TRANSFER] Engine created: dp_rank=%d gpu=%d "
            "address=%s",
            dp_rank,
            local_gpu_id,
            self._nccl_engine.zmq_address,
        )

        # Pre-establish NCCL connection to the home GPU to avoid
        # 100-200ms setup cost on the first transfer.
        import os
        home_rank = int(os.environ.get("SAGE_HOME_GPU", "0"))
        if dp_rank != home_rank:
            from vllm.utils.network_utils import get_ip
            home_port = int(kv_transfer_config.kv_port) + home_rank
            home_address = f"{get_ip()}:{home_port}"
            logger.info(
                "[SAGE_KV_TRANSFER] Pre-connecting to home at %s",
                home_address,
            )
            self._nccl_engine.create_connect(home_address)
            logger.info(
                "[SAGE_KV_TRANSFER] Pre-connected to home",
            )

    @property
    def address(self) -> str:
        """This engine's ZMQ address (for embedding in chunk requests)."""
        return self._nccl_engine.zmq_address

    def set_kv_caches(self, kv_caches: list[torch.Tensor]) -> None:
        """Set the paged KV cache tensors (one per layer)."""
        self._kv_caches = kv_caches
        self._num_layers = len(kv_caches)
        if self._num_layers > 0:
            first = kv_caches[0]
            self._is_flash_attn = first.dim() >= 3 and first.shape[0] == 2
        _layout = "flash [2,blocks,bsz,H,D]" if self._is_flash_attn else "triton/other [blocks,2,bsz,H,D]"
        logger.info(
            "[SAGE_KV_TRANSFER] KV caches set: %d layers, layout=%s shape=%s",
            self._num_layers,
            _layout,
            tuple(kv_caches[0].shape) if self._num_layers > 0 else "N/A",
        )

    def set_layer_names(self, layer_names: list[str]) -> None:
        """Set attention layer names for per-layer receive."""
        self._layer_names = layer_names
        logger.info(
            "[SAGE_KV_TRANSFER] Layer names set: %d layers",
            len(layer_names),
        )

    def _inject_kv(
        self, layer: torch.Tensor, kv_data: torch.Tensor,
        block_ids: torch.Tensor,
    ) -> None:
        """Inject KV into a single layer's paged cache."""
        if self._is_flash_attn:
            layer[:, block_ids, ...] = kv_data
        else:
            layer[block_ids, ...] = kv_data

    def _extract_kv(
        self, layer: torch.Tensor, block_ids: torch.Tensor
    ) -> torch.Tensor:
        """Extract KV from a single layer's paged cache."""
        if self._is_flash_attn:
            return layer[:, block_ids, ...].contiguous()
        return layer[block_ids, ...].contiguous()

    def send_layer(
        self,
        layer_idx: int,
        layer_name: str,
        block_ids: torch.Tensor,
        dest_address: str,
        transfer_id: str,
    ) -> None:
        """Send a single layer's KV for a chunk to the home GPU.

        Called from the per-layer send callback during model forward.
        The tensor_id format matches recv_chunk_per_layer_payload's
        recv_tensor call: f"{transfer_id}#{layer_name}".
        """
        if self._kv_caches is None:
            raise RuntimeError("KV caches not set")
        kv_data = self._extract_kv(self._kv_caches[layer_idx], block_ids)
        self._nccl_engine.send_tensor(
            f"{transfer_id}#{layer_name}", kv_data, dest_address,
        )

    def send_chunk(
        self,
        block_ids: list[int],
        token_ids: list[int],
        dest_address: str,
        transfer_id: str,
        image_grid_thw: list[list[int]] | None = None,
        total_chunks: int = 0,
        max_tokens: int = 1,
        min_tokens: int = 0,
        query_token_count: int = 0,
    ) -> None:
        """Send a chunk's metadata + bulk KV to a peer.

        Protocol (2 NCCL sends):
          1. metadata: [num_blocks, num_images, total_chunks, max_tokens,
                        min_tokens, query_token_count,
                        mm_t0,h0,w0,..., tok_0, tok_1, ...]
          2. bulk_kv: all layers concatenated
        """
        if self._kv_caches is None:
            raise RuntimeError("KV caches not set — call set_kv_caches first")
        block_ids_t = torch.tensor(
            block_ids, dtype=torch.long, device=self.device
        )
        num_images = len(image_grid_thw) if image_grid_thw else 0
        meta_list: list[int] = [
            len(block_ids), num_images, total_chunks,
            max_tokens, min_tokens, query_token_count,
        ]
        if image_grid_thw:
            for thw in image_grid_thw:
                meta_list.extend(thw)
        meta_list.extend(token_ids)
        metadata = torch.tensor(
            meta_list, dtype=torch.long, device=self.device,
        )
        self._nccl_engine.send_tensor(
            f"{transfer_id}#metadata", metadata, dest_address
        )
        kv_layers = [
            self._extract_kv(self._kv_caches[i], block_ids_t)
            for i in range(self._num_layers)
        ]
        bulk_kv = torch.cat(
            kv_layers, dim=0 if not self._is_flash_attn else 1
        )
        self._nccl_engine.send_tensor(
            f"{transfer_id}#bulk_kv", bulk_kv, dest_address
        )
        logger.info(
            "[SAGE_KV_TRANSFER] Sent chunk (%d tokens, %d blocks, "
            "%d layers, %d images, bulk) to %s [id=%s]",
            len(token_ids), len(block_ids), self._num_layers,
            num_images, dest_address, transfer_id,
        )

    def recv_chunk(
        self,
        transfer_id: str,
        allocate_blocks_fn: "Callable[[int], list[int]]",
        timeout: float = 300.0,
    ) -> dict:
        """Receive a chunk's metadata + bulk KV.

        Returns dict with keys:
            token_ids, dest_block_ids, num_tokens, num_blocks,
            image_grid_thw, total_chunks, max_tokens, min_tokens,
            query_token_count
        """
        if self._kv_caches is None:
            raise RuntimeError("KV caches not set — call set_kv_caches first")
        metadata = self._nccl_engine.recv_tensor(
            f"{transfer_id}#metadata", timeout=timeout,
        )
        meta_cpu = metadata.cpu().tolist()
        num_blocks = int(meta_cpu[0])
        num_images = int(meta_cpu[1])
        total_chunks = int(meta_cpu[2])
        max_tokens = int(meta_cpu[3])
        min_tokens = int(meta_cpu[4])
        query_token_count = int(meta_cpu[5])
        # Parse mm metadata: 3 ints per image after the 6-int header
        image_grid_thw: list[list[int]] = []
        mm_end = 6
        for i in range(num_images):
            offset = 6 + i * 3
            image_grid_thw.append(
                [int(meta_cpu[offset]), int(meta_cpu[offset + 1]),
                 int(meta_cpu[offset + 2])]
            )
            mm_end = offset + 3
        token_ids = [int(t) for t in meta_cpu[mm_end:]]
        num_tokens = len(token_ids)

        dest_block_ids = allocate_blocks_fn(num_blocks)
        dest_ids_t = torch.tensor(
            dest_block_ids, dtype=torch.long, device=self.device
        )
        bulk_kv = self._nccl_engine.recv_tensor(
            f"{transfer_id}#bulk_kv", timeout=timeout,
        )
        if self._is_flash_attn:
            chunks = torch.chunk(bulk_kv, self._num_layers, dim=1)
        else:
            chunks = torch.chunk(bulk_kv, self._num_layers, dim=0)
        for layer_idx, kv_data in enumerate(chunks):
            self._inject_kv(self._kv_caches[layer_idx], kv_data, dest_ids_t)
        logger.info(
            "[SAGE_KV_TRANSFER] Received chunk (%d tokens, %d blocks, "
            "%d layers, %d images, bulk) [id=%s]",
            num_tokens, num_blocks, self._num_layers,
            num_images, transfer_id,
        )
        return {
            "token_ids": token_ids,
            "dest_block_ids": dest_block_ids,
            "num_tokens": num_tokens,
            "num_blocks": num_blocks,
            "image_grid_thw": image_grid_thw,
            "total_chunks": total_chunks,
            "max_tokens": max_tokens,
            "min_tokens": min_tokens,
            "query_token_count": query_token_count,
        }

    def recv_chunk_per_layer_header(self, transfer_id: str) -> dict:
        """Receive only the per-layer chunk's metadata header.

        Header layout (must match EngineCore early header send):
            [num_tokens, num_blocks, total_chunks, max_tokens, min_tokens,
             query_token_count, tok_0, tok_1, ...]
        """
        if not hasattr(self, "_layer_names") or not self._layer_names:
            raise RuntimeError("Layer names not set — call set_layer_names")
        header = self._nccl_engine.recv_tensor(f"{transfer_id}#header")
        header_cpu = header.cpu().tolist()
        return {
            "num_tokens": int(header_cpu[0]),
            "num_blocks": int(header_cpu[1]),
            "total_chunks": int(header_cpu[2]),
            "max_tokens": int(header_cpu[3]),
            "min_tokens": int(header_cpu[4]),
            "query_token_count": int(header_cpu[5]),
            "token_ids": [int(t) for t in header_cpu[6:]],
        }

    def recv_chunk_per_layer_payload(
        self,
        transfer_id: str,
        dest_block_ids: list[int],
        on_layer_done: "Callable[[int], None] | None" = None,
    ) -> None:
        """Receive all per-layer KV tensors for a chunk and inject them
        into the pre-allocated destination blocks.

        Runs from a background Python thread; uses the default CUDA
        stream because the actual benefit on a single-node SAGE setup
        comes from the early-header send + pipelined-blend trigger,
        not from stream-level parallelism (which is bounded by
        cross-process Hyper-Q on the GPU anyway).
        """
        if self._kv_caches is None:
            raise RuntimeError("KV caches not set")
        if not hasattr(self, "_layer_names") or not self._layer_names:
            raise RuntimeError("Layer names not set — call set_layer_names")
        dest_ids_t = torch.tensor(
            dest_block_ids, dtype=torch.long, device=self.device
        )
        for layer_idx in range(self._num_layers):
            layer_name = self._layer_names[layer_idx]
            _tid = f"{transfer_id}#{layer_name}"
            if layer_idx == 0:
                logger.info(
                    "[SAGE_KV_TRANSFER] Drain start %s (waiting for L0)",
                    transfer_id,
                )
            try:
                kv_data = self._nccl_engine.recv_tensor(
                    _tid, timeout=30.0,
                )
            except TimeoutError:
                logger.error(
                    "[SAGE_KV_TRANSFER] TIMEOUT waiting for %s "
                    "(layer %d/%d)", _tid, layer_idx, self._num_layers,
                )
                raise
            self._inject_kv(
                self._kv_caches[layer_idx], kv_data, dest_ids_t,
            )
            if on_layer_done is not None:
                on_layer_done(layer_idx)
        # Ensure all inject kernels have completed on the GPU before
        # signaling "drain done".
        torch.cuda.synchronize(self.device)
        logger.info(
            "[SAGE_KV_TRANSFER] Drained per-layer payload (%d layers, "
            "%d blocks) [id=%s]",
            self._num_layers, len(dest_block_ids), transfer_id,
        )

    def recv_chunk_per_layer(
        self,
        transfer_id: str,
        allocate_blocks_fn: "Callable[[int], list[int]]",
    ) -> dict:
        """Synchronous per-layer recv: header → allocate → drain payload.

        Kept for callers that want the legacy single-threaded path.
        """
        if self._kv_caches is None:
            raise RuntimeError("KV caches not set")
        if not hasattr(self, "_layer_names") or not self._layer_names:
            raise RuntimeError("Layer names not set — call set_layer_names")

        meta = self.recv_chunk_per_layer_header(transfer_id)
        num_blocks = meta["num_blocks"]
        dest_block_ids = allocate_blocks_fn(num_blocks)
        self.recv_chunk_per_layer_payload(transfer_id, dest_block_ids)
        return {
            "token_ids": meta["token_ids"],
            "dest_block_ids": dest_block_ids,
            "num_tokens": meta["num_tokens"],
            "num_blocks": num_blocks,
            "image_grid_thw": [],
            "total_chunks": meta["total_chunks"],
            "max_tokens": meta["max_tokens"],
            "min_tokens": meta["min_tokens"],
            "query_token_count": meta["query_token_count"],
        }
