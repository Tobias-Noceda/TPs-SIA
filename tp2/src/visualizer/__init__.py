import json
import sys

from typing import Callable

import pygame
from multiprocessing import shared_memory

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

BACKGROUND_COLOR = WHITE

WIDTH = 500
HEIGHT = 500

type State = dict

type OnUpdate = Callable[[], State | None]

def open_shm(name: str) -> shared_memory.SharedMemory:
    return shared_memory.SharedMemory(name)

def get_shm_state(shm: shared_memory.SharedMemory):
    if shm.buf is None:
        raise BufferError("Shared memory buffer is closed")

    length = int.from_bytes(shm.buf[:4], "little")

    if length == 0:
        return None

    data = bytes(shm.buf[4:4 + length])
    return json.loads(data.decode("utf-8"))

def draw(screen: pygame.Surface, state: State | None) -> None:
    global WIDTH, HEIGHT

    screen.fill(BACKGROUND_COLOR)

    if state is not None:
        if "size" in state and tuple(state["size"]) != (WIDTH, HEIGHT):
            WIDTH, HEIGHT = state["size"]
            pygame.display.set_mode((WIDTH, HEIGHT))

        for polygon in state["polygons"]:
            pygame.draw.polygon(screen, polygon["color"], polygon["vertices"])

def loop(update: OnUpdate) -> None:
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    running = True
    state = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        try:
            new_state = update()
            if new_state is not None:
                state = new_state
        except:
            pass

        draw(screen, state)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

def main() -> None:
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            data = json.load(f)

        loop(lambda: data)
        return

    try:
        shm = open_shm("visualizer_state")
    except:
        print("Backend shared memory does not exist.")
        return

    loop(lambda: get_shm_state(shm))
