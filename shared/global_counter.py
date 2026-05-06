"""Cluster-wide atomic counters accessible from any Ray node.

Used to track in-flight request states across the training pipeline.
The counter is a detached Ray actor so it survives job restarts.

Usage:
    from shared.global_counter import get_global_counter, counter_scope

    # Manual inc/dec
    counter = get_global_counter()
    ray.get(counter.inc.remote("my_key"))
    ray.get(counter.dec.remote("my_key"))

    # Context manager (fire-and-forget, no await overhead)
    async with counter_scope("assistant_generate"):
        result = await do_generation(...)
"""

import asyncio
import logging
import threading
import time

import httpx
import ray
import wandb

logger = logging.getLogger(__name__)

_ACTOR_NAME = "tau2_global_counter"


@ray.remote(num_cpus=0)
class GlobalCounter:
    def __init__(self):
        self._counts: dict[str, int] = {}

    def inc(self, key: str) -> None:
        self._counts[key] = self._counts.get(key, 0) + 1

    def dec(self, key: str) -> None:
        self._counts[key] = self._counts.get(key, 0) - 1

    def get_all(self) -> dict[str, int]:
        return dict(self._counts)


_counter_lock = threading.Lock()
_enabled = False


def enable_global_counter():
    """Enable the global counter. Must be called explicitly during training init.

    The counter is disabled by default so that evaluation and other code paths
    that import counter_scope don't accidentally create the Ray actor.
    """
    global _enabled
    _enabled = True


def get_global_counter() -> ray.actor.ActorHandle:
    """Get (or create) the cluster-wide GlobalCounter named actor."""
    try:
        return ray.get_actor(_ACTOR_NAME)
    except ValueError:
        with _counter_lock:
            try:
                return ray.get_actor(_ACTOR_NAME)
            except ValueError:
                return GlobalCounter.options(
                    name=_ACTOR_NAME,
                    lifetime="detached",
                ).remote()


class counter_scope:
    """Async context manager that increments on enter, decrements on exit.

    No-op if the global counter has not been enabled via enable_global_counter().
    Uses fire-and-forget calls (no await) to avoid adding latency.
    """

    def __init__(self, key: str):
        self.key = key

    async def __aenter__(self):
        if _enabled:
            get_global_counter().inc.remote(self.key)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if _enabled:
            get_global_counter().dec.remote(self.key)
        return False


async def log_counter_loop(interval: float = 10.0, router_url: str | None = None):
    """Periodically log all counter values and send to wandb.

    Logs to both Python logger and wandb (if available).
    The wandb x-axis is wall-clock seconds since the loop started,
    using a custom step metric to avoid conflicting with the training step.

    If *router_url* is provided, also queries the router's ``/workers``
    endpoint, then each worker's ``/get_load`` and ``/get_server_info``
    to log running requests, queued requests, tokens, and throughput.
    """
    counter = get_global_counter()
    start_time = time.monotonic()

    router_client: httpx.AsyncClient | None = None
    if router_url:
        router_client = httpx.AsyncClient(timeout=httpx.Timeout(5.0))
    seq = 0
    while True:
        await asyncio.sleep(interval)
        try:
            counts = ray.get(counter.get_all.remote())

            # Query router for workers, then each worker for load metrics
            if router_client is not None:
                try:
                    resp = await router_client.get(f"{router_url}/workers")
                    resp.raise_for_status()
                    workers = resp.json().get("workers", [])

                    total_running = 0
                    total_queued = 0
                    total_tokens = 0
                    total_throughput = 0.0
                    num_healthy = 0

                    for worker in workers:
                        url = worker["url"]
                        is_healthy = worker.get("is_healthy", False)
                        if is_healthy:
                            num_healthy += 1

                        # Get load (running/queued/tokens)
                        try:
                            load_resp = await router_client.get(f"{url}/get_load")
                            load_resp.raise_for_status()
                            load_data = load_resp.json()
                            if isinstance(load_data, list) and load_data:
                                running = sum(d.get("num_reqs", 0) for d in load_data)
                                queued = sum(d.get("num_waiting_reqs", 0) for d in load_data)
                                tokens = sum(d.get("num_tokens", 0) for d in load_data)
                            elif isinstance(load_data, dict):
                                running = load_data.get("num_reqs", 0)
                                queued = load_data.get("num_waiting_reqs", 0)
                                tokens = load_data.get("num_tokens", 0)
                            else:
                                running = queued = tokens = 0
                            total_running += running
                            total_queued += queued
                            total_tokens += tokens
                        except Exception:
                            pass

                        # Get throughput
                        try:
                            info_resp = await router_client.get(f"{url}/get_server_info")
                            info_resp.raise_for_status()
                            info = info_resp.json()
                            internal = info.get("internal_states", [])
                            if internal:
                                total_throughput += internal[0].get("last_gen_throughput", 0)
                        except Exception:
                            pass

                    counts["router_total_running"] = total_running
                    counts["router_total_queued"] = total_queued
                    counts["router_total_tokens"] = total_tokens
                    counts["router_total_throughput"] = total_throughput
                    counts["router_num_workers"] = len(workers)
                    counts["router_num_healthy"] = num_healthy
                except Exception as e:
                    logger.debug("[global-counter] Failed to query router metrics: %s", e, exc_info=True)

            if counts:
                elapsed = time.monotonic() - start_time
                parts = [f"{k}={v}" for k, v in sorted(counts.items())]
                seq += 1
                logger.info(f"[global-counter #{seq} t={elapsed:.0f}s] {' | '.join(parts)}")

                log_data = {f"counter/{k}": v for k, v in counts.items()}
                log_data["counter/_elapsed_s"] = elapsed
                wandb.log(log_data)
        except Exception as e:
            logger.exception(f"[global-counter] Error in log_counter_loop: {e}")
            pass
