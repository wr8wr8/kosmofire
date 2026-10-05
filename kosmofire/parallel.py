import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


PER_WORKER_MEMORY_BYTES = 2 * 1024**3


def available_memory_bytes() -> int | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None
    return int(pages) * int(page_size)


def default_workers() -> int:
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 1
    memory = available_memory_bytes()
    by_memory = max(1, int(memory // PER_WORKER_MEMORY_BYTES)) if memory else cpus
    return max(1, min(64, cpus, by_memory))


def parallel_map(func: Callable[[T], R], items: Iterable[T], workers: int | None = None) -> list[R]:
    items = list(items)
    workers = workers or default_workers()
    if workers == 1 or len(items) < 2:
        return [func(item) for item in items]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(func, items, chunksize=1))
