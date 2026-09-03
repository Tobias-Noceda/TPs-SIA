import json
import sys

import pygame
from multiprocessing import shared_memory

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

BACKGROUND_COLOR = WHITE

WIDTH = 500
HEIGHT = 500

def open_shm(name: str) -> shared_memory.SharedMemory:
    return shared_memory.SharedMemory(name)

def read_state(shm: shared_memory.SharedMemory):
    if shm.buf is None:
        raise BufferError("Shared memory buffer is closed")

    length = int.from_bytes(shm.buf[:4], "little")

    if length == 0:
        return None

    data = bytes(shm.buf[4:4 + length])
    return json.loads(data.decode("utf-8"))

def loop(shm_name: str) -> None:
    try:
        shm = open_shm(shm_name)
    except:
        print("Backend shared memory does not exist.")
        return

    pygame.init()

    screen = pygame.display.set_mode([WIDTH, HEIGHT])
    clock = pygame.time.Clock()

    running = True
    state = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        try:
            new_state = read_state(shm)

            if new_state is not None:
                state = new_state
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

        screen.fill(BACKGROUND_COLOR)

        if state is not None:
            for triangle in state["triangles"]:
                a = tuple(triangle["a"])
                b = tuple(triangle["b"])
                c = tuple(triangle["c"])

                color = triangle["color"]

                pygame.draw.polygon(screen, color, (a, b, c))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


def main() -> None:
    loop(sys.argv[1] if len(sys.argv) > 1 else "visualizer_state")
