import os

# Environment of the sweep workers. It must be set in the parent before the first job is submitted:
# the workers are started from it (forkserver), so they inherit it before importing numpy and tqdm,
# which read these variables only once, at import time.
_BLAS_THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")

def prepare_workers(workers: int) -> None:
    """
    Splits the cores among the worker processes and silences their progress bars.

    Without a limit every worker's BLAS opens one thread per core: with N workers there are N × cores
    threads fighting for the cores and training gets ~20x slower. The bars of parallel workers would
    also overwrite each other on the same line.
    """
    threads = max(1, (os.cpu_count() or 1) // max(1, workers))
    for name in _BLAS_THREADS:
        os.environ[name] = str(threads)
    os.environ["TQDM_DISABLE"] = "1"
