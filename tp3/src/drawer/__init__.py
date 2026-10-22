import pygame
import numpy as np
from tp3 import create_model
from tp3.config import load_config

GRID_SIZE = 28
PIXEL_SIZE = 20

CANVAS_SIZE = GRID_SIZE * PIXEL_SIZE

WINDOW_WIDTH = CANVAS_SIZE
WINDOW_HEIGHT = CANVAS_SIZE

BG_COLOR = (0, 0, 0)
GRID_COLOR = (60, 60, 60)
WHITE = (255, 255, 255)

MODEL, *_ = create_model(load_config("res/config/config_best_digits.json"))

pygame.init()

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("28x28 Number Analyzer")

pixels = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)

clock = pygame.time.Clock()

drawing = True

def draw_canvas():
    screen.fill(BG_COLOR)

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            value = int(pixels[y, x] * 255)
            color = (value, value, value)

            rect = pygame.Rect(
                x * PIXEL_SIZE,
                y * PIXEL_SIZE,
                PIXEL_SIZE,
                PIXEL_SIZE
            )

            pygame.draw.rect(
                screen,
                color,
                rect
            )

            pygame.draw.rect(
                screen,
                GRID_COLOR,
                rect,
                1
            )

def paint(mouse_x, mouse_y):
    x = mouse_x // PIXEL_SIZE
    y = mouse_y // PIXEL_SIZE

    if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
        if drawing:
            pixels[y, x] = 1.0
        else:
            pixels[y, x] = 0.0

def clear():
    pixels.fill(0)

def get_model_prediction():
    image = pixels.copy().reshape(1, -1)
    predictions = MODEL.predict(image)[0]
    return np.argmax(predictions), predictions

def main():
    global drawing

    while True:
        painted = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    drawing = True
                    paint(*event.pos)
                    painted = True

                elif event.button == 3:
                    drawing = False
                    paint(*event.pos)
                    painted = True

            elif event.type == pygame.MOUSEMOTION:
                if pygame.mouse.get_pressed()[0]:
                    drawing = True
                    paint(*event.pos)
                    painted = True

                elif pygame.mouse.get_pressed()[2]:
                    drawing = False
                    paint(*event.pos)
                    painted = True

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    clear()

            if painted:
                n, p = get_model_prediction()
                print(f"That's a {n} with {float(p[n]):.2f} confidence")

        draw_canvas()

        # font = pygame.font.Font(None, 24)

        # text = font.render("Left: Draw   Right: Erase", True, WHITE)
        # screen.blit(text, (10, CANVAS_SIZE + 20))

        pygame.display.flip()
        clock.tick(60)
