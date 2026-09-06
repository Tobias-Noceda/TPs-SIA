import json
from multiprocessing import shared_memory

from tp2.types.coords import Coords
from tp2.types.individual import Individual

SHM_NAME = "visualizer_state"
SHM_SIZE = 4096

class RealTimeVis:
    size: Coords
    bg: tuple[int, ...]

    shm: shared_memory.SharedMemory

    def __init__(self, size: Coords, bg: tuple[int, ...]) -> None:
        self.size = size
        self.bg = bg

    def __enter__(self):
        try:
            self.shm = shared_memory.SharedMemory(
                name=SHM_NAME,
                create=True,
                size=SHM_SIZE,
            )
        except FileExistsError:
            old_shm = shared_memory.SharedMemory(name=SHM_NAME)
            old_shm.unlink()
            old_shm.close()

            self.shm = shared_memory.SharedMemory(
                name=SHM_NAME,
                create=True,
                size=SHM_SIZE,
            )

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shm.close()
        self.shm.unlink()

    def update(self, individual: Individual):
        polygons = [
            {
                "vertices": triangle.vertices,
                "color": triangle.color.pillow()
            }
            for triangle in individual
        ]

        state = {
            "counter": 0,
            "size": self.size,
            "background": self.bg,
            "polygons": polygons,
        }

        self._write_shm(state)

    def _write_shm(self, state: dict):
        data = json.dumps(state).encode("utf-8")

        if len(data) > SHM_SIZE:
            raise ValueError("State is too large for shared memory")

        if self.shm.buf is None:
            raise BufferError("Shared memory buffer is closed")

        self.shm.buf[:4] = len(data).to_bytes(4, "little")
        self.shm.buf[4:4 + len(data)] = data
