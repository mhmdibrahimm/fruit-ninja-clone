import os
import math
import numpy as np
import pygame
import pgzrun
import random
import pandas as pd
import tensorflow as tf
from actors import *
from typing import List
from pgzero.rect import Rect
from pgzero.screen import Screen
from screeninfo import get_monitors

from helpers import load, save

# --- Game Constants ---
FRAME_RATE = 60  # target fps
BASE_SPAWN_INTERVAL = FRAME_RATE * 1.5  # base delay between fruit spawns
WHITE_FLASH_DURATION = FRAME_RATE * 2  # frames for white flash on bomb hit
RED_STATIC_DURATION = FRAME_RATE * 1  # frames for red static screen after death
ZOOM_DURATION = FRAME_RATE / 4  # frames for zoom‑out “Game Over” animation
GRAVITY: float = 0.2  # gravity acceleration applied per frame
GANG_OF_THREE_FONT = "go3v2"  # font name used for all on‑screen text
BASE_VX_RANGE = (-3, 3)  # horizontal velocity range for new fruits
BASE_VY_RANGE = (-18, -14)  # upward velocity range for new fruits
MAX_TRAIL_STEPS = 5  # max steps for cursor trail
PERF_WINDOW_SIZE = 20  # number of recent records used for AI difficulty

# --- Difficulty Presets ---
DIFFICULTIES = {
    "easy": dict(spawn_mult=0.8, max_count=2, bomb_chance=0.1),
    "normal": dict(spawn_mult=1.0, max_count=3, bomb_chance=0.3),
    "hard": dict(spawn_mult=1.0, max_count=4, bomb_chance=0.5),
    "impossible": dict(spawn_mult=1.0, max_count=10, bomb_chance=0.9),
}
MODE_AI = "ai_detect"  # key for selecting AI‑adaptive difficulty mode

# --- Global State ---
current_screen: str = "menu"  # "menu", "game", "paused", or "gameover"
mouse_held: bool = False  # True while the slice button is held down
screen: Screen  # main Pygame Zero Screen object (initiated for IDE type-check)
performance_history: List[bool] = []  # success/failure history for AI training
performance_data: List[dict] = []  # performance records to append to CSV
session_attempts: int = 0  # count of completed game sessions
selected_difficulty: str = load()[1]  # user's chosen difficulty
best_score: int = int(load()[0])  # highest score across games

# --- Cursor Trail ---
cursor_trail: List[tuple] = []  # list of (x, y, life) tuples for trail effect
last_trail_position = None  # previous mouse position for trail interpolation

# --- Game Stats ---
score: int = 0  # current round’s slice score
missed: int = 0  # number of fruits dropped without slicing

# --- Gameover Variables ---
gameover_phase: int = 0  # sub‑phase of the gameover animation sequence
gameover_timer: int = 0  # frame countdown for the current gameover phase
gameover_by_bomb: bool = False  # True if the death was caused by slicing a bomb

# --- Fruits and Splashes ---
fruits: List[Fruit] = []  # active fruit and bomb objects on screen
splashes: List[Splash] = []  # active splash effect objects on screen
spawn_interval: int = BASE_SPAWN_INTERVAL  # dynamic interval between spawns
spawn_timer: int = spawn_interval  # countdown to next spawn

# --- AI Model Setup ---
MODEL_PATH = "ai_model.weights.h5"  # file path for saving/loading weights
model = None  # TensorFlow model instance for AI difficulty
predicted_index: int = None  # a


# Initializes, trains, and saves a simple Keras model to predict an appropriate difficulty level based on saved past performance data.
def build_and_train_model():
    global model
    if not os.path.isfile("performance_data.csv"):
        print("AI Detect: no data file found—skipping model build.")
        return None
    df = pd.read_csv("performance_data.csv")
    if df.empty:
        print("AI Detect: data file empty—skipping model build.")
        return None

    X = df[["accuracy", "reaction_time", "attempts"]].values
    y = df["level"].astype(float).values

    # define & train
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(16, activation="relu", input_shape=(3,)),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=20, verbose=0)
    model.save_weights(MODEL_PATH)
    print("AI Detect: model trained & weights saved.")
    return model


model = build_and_train_model()


# Computes the player’s recent success rate for use in difficulty.
def get_difficulty_factor() -> float:
    if not performance_history:
        return 0.5
    recent = performance_history[-PERF_WINDOW_SIZE:]
    return sum(recent) / len(recent)


# Appends performance records and clears the in‑memory buffer.
def save_performance_data():
    if not performance_data:
        return

    df_new = pd.DataFrame(performance_data)
    fname = "performance_data.csv"
    file_exists = os.path.isfile(fname)

    df_new.to_csv(fname, mode="a", header=not file_exists, index=False)

    performance_data.clear()

    print("AI Detect: performance_data.csv appended with", len(df_new), "records.")


# Renders a filled circle with variable transparency at a given position.
def draw_transparent_circle(pos, radius, color, alpha):
    surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    pygame.draw.circle(surf, color + (alpha,), (radius, radius), radius)
    screen.surface.blit(surf, (pos[0] - radius, pos[1] - radius))


# Transitions the game into its “game over” sequence.
def trigger_gameover(bomb_triggered: bool = False):
    global current_screen, gameover_phase, gameover_timer, gameover_by_bomb, model
    current_screen = "gameover"
    gameover_by_bomb = bomb_triggered
    gameover_phase = 0 if bomb_triggered else 1
    gameover_timer = WHITE_FLASH_DURATION if bomb_triggered else RED_STATIC_DURATION


# Adds a wave of new fruit to the game.
def spawn_fruits(diff: float, preset: dict):
    global fruits
    base = random.randint(1, preset["max_count"])
    extra = (
        1
        if (selected_difficulty == MODE_AI and diff > 0.8 and random.random() < diff)
        else 0
    )
    count = min(base + extra, preset["max_count"])
    bomb_incl = count > 1 and random.random() < preset["bomb_chance"]
    bomb_idx = random.randrange(count) if bomb_incl else -1

    for i in range(count):
        name = (
            "bomb"
            if i == bomb_idx
            else random.choice(["watermelon", "apple_green", "apple_red"])
        )
        x, y = random.randint(100, WIDTH - 100), HEIGHT + 50
        vx, vy = random.uniform(*BASE_VX_RANGE), random.uniform(*BASE_VY_RANGE)
        fruits.append(TrackedFruit(name, i == bomb_idx, (x, y), vx, vy))


# Resets all statistics.
def reset_game():
    global score, missed, spawn_timer, fruits, splashes, session_attempts
    score, missed = 0, 0
    spawn_timer = BASE_SPAWN_INTERVAL
    fruits.clear()
    splashes.clear()
    session_attempts += 1


# Central update handler called by pgzero on each new frame: advances game objects, checks for collisions/misses, handles spawning and gameover logic, and updates cursor trail.
def update():
    global model, spawn_timer, missed, score, best_score, spawn_interval, predicted_index, selected_difficulty, gameover_phase, gameover_timer

    if current_screen == "game":
        if (
            selected_difficulty == MODE_AI
            and model is not None
            and len(performance_data) >= PERF_WINDOW_SIZE
        ):
            recent = pd.DataFrame(performance_data[-PERF_WINDOW_SIZE:])
            acc = recent["accuracy"].mean()
            rt = recent["reaction_time"].mean()
            feats = np.array([[acc, rt, session_attempts]])
            pred = model.predict(feats, verbose=0)[0][0]
            lvl = int(np.clip(round(pred), 0, len(DIFFICULTIES) - 1))
            predicted_index = lvl

        for f in fruits[:]:
            f.updateActor(GRAVITY)
            if f.actor.y > HEIGHT + 50 and (f.actor.x >= 0 and f.actor.x <= WIDTH):
                fruits.remove(f)
                if not f.getSliced() and not f.is_bomb:
                    missed += 1
                    performance_history.append(False)
                    performance_data.append(
                        {
                            "accuracy": 0.0,
                            "reaction_time": 0.0,
                            "attempts": session_attempts,
                            "level": (
                                4
                                if selected_difficulty == MODE_AI
                                else list(DIFFICULTIES.keys()).index(
                                    selected_difficulty
                                )
                            ),
                        }
                    )
                    if missed >= 3:
                        if score > best_score:
                            best_score = score
                            save(str(best_score))
                        trigger_gameover()
                        break

        spawn_timer -= 1
        if spawn_timer <= 0:
            preset = (
                DIFFICULTIES[selected_difficulty]
                if selected_difficulty != MODE_AI
                else DIFFICULTIES["normal"]
            )
            diff = get_difficulty_factor() if selected_difficulty == MODE_AI else 0.0
            spawn_interval = int(
                BASE_SPAWN_INTERVAL * preset["spawn_mult"] * (1.0 - 0.3 * diff)
            )
            spawn_fruits(diff, preset)
            spawn_timer = spawn_interval

    elif current_screen == "gameover":
        if gameover_timer > 0:
            gameover_timer -= 1
        else:
            if gameover_phase == 0:
                gameover_phase = 1
                gameover_timer = RED_STATIC_DURATION
            elif gameover_phase == 1:
                gameover_phase = 2
                gameover_timer = ZOOM_DURATION
            elif gameover_phase == 2:
                gameover_phase = 3
                save_performance_data()
                model_update = build_and_train_model()
                if model_update:
                    model = model_update
    new_trail = []
    for x, y, l in cursor_trail:
        l -= 1
        if l > 0:
            new_trail.append((x, y, l))
    cursor_trail[:] = new_trail


# Renders the main menu, difficulty selectors, and disclaimer.
def draw_menu_screen():
    global selected_difficulty
    bg = pygame.image.load("images/menu_background.png")
    bg_rect = bg.get_rect()
    bg_rect.center = (WIDTH // 2, HEIGHT // 2)
    screen.blit("menu_background", bg_rect.topleft)
    screen.draw.text(
        f"This project incorporates material from the original game, which is the copyrighted property of its respective owners.\nAll rights remain with the original creators.\nThis work is intended solely for educational purposes and is not for commercial use.",
        center=(WIDTH / 2, 50),
        fontsize=15,
        color="white",
        fontname=GANG_OF_THREE_FONT,
        owidth=2,
        ocolor="black",
    )
    logo = pygame.image.load("images/menu_logo.png")
    logo_rect = logo.get_rect()
    logo_rect.center = (WIDTH // 2, HEIGHT // 2)
    screen.blit("menu_logo", logo_rect.topleft)
    start_rect = Rect((WIDTH / 2 - 175, HEIGHT - 200), (350, 122))
    screen.blit("button_bg", start_rect.topleft)
    screen.draw.text(
        "Start Game",
        center=start_rect.center,
        fontsize=50,
        color="#5A171E",
        fontname=GANG_OF_THREE_FONT,
    )
    for i, (name, _) in enumerate((DIFFICULTIES | {MODE_AI: None}).items()):
        x = 100
        y = ((HEIGHT / 2) - 53) + (i * 60)
        rect = Rect((x - 60, y - 30), (151, 53))
        screen.blit(
            "button_small"
            + (
                "_yellow"
                if name != selected_difficulty
                else ("_red" if name == "impossible" else "_green")
            ),
            rect.topleft,
        )
        screen.draw.text(
            name.capitalize(),
            center=rect.center,
            fontsize=25,
            color="#5A171E",
            fontname=GANG_OF_THREE_FONT,
        )


# Renders the in‑game HUD: score, best score, misses, pause button, and splashes/fruits.
def draw_game_screen():
    screen.blit("resume", (80, HEIGHT - 100))
    for splash in splashes:
        splash.actor.draw()
    for fruit in fruits:
        fruit.actor.draw()
    screen.blit("score_watermelon", (80, 40))

    screen.draw.text(
        f"{score}",
        topleft=(170, 40),
        fontsize=75,
        color="#E79435",
        fontname=GANG_OF_THREE_FONT,
        owidth=2,
        ocolor="black",
    )
    screen.draw.text(
        f"Best: {best_score}",
        topleft=(80, 140),
        fontsize=40,
        color="#E79435",
        fontname=GANG_OF_THREE_FONT,
        owidth=2,
        ocolor="black",
    )
    if selected_difficulty == MODE_AI and predicted_index is not None:
        screen.draw.text(
            f"AI Level → {predicted_index}",
            topleft=(80, 200),
            fontsize=30,
            color="white",
            fontname=GANG_OF_THREE_FONT,
            owidth=2,
            ocolor="black",
        )
    screen.blit(f"missed_{0 if missed > 3 else missed}", (WIDTH - 350, 50))


# Animates the multi‑phase game‑over sequence: white flash, static “Game Over,” zoom, then full summary.
def draw_gameover_screen():
    if gameover_phase == 0 and gameover_by_bomb:
        progress = (WHITE_FLASH_DURATION - gameover_timer) / WHITE_FLASH_DURATION
        current_alpha = 153 + progress * (255 - 153)
        white_overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        white_overlay.fill((255, 255, 255, int(current_alpha)))
        screen.surface.blit(white_overlay, (0, 0))
    elif gameover_phase == 1:
        screen.draw.text(
            "Game Over",
            center=(WIDTH / 2, HEIGHT / 2),
            fontsize=120,
            color="red",
            fontname=GANG_OF_THREE_FONT,
            owidth=3,
            ocolor="black",
        )
    elif gameover_phase == 2:
        progress = 1 - (gameover_timer / ZOOM_DURATION)
        size = 120 - progress * (120 - 60)
        screen.draw.text(
            "Game Over",
            center=(WIDTH / 2, HEIGHT / 2),
            fontsize=int(size),
            color="red",
            fontname=GANG_OF_THREE_FONT,
            owidth=3,
            ocolor="black",
        )
    elif gameover_phase == 3:
        y_game_over = 150
        y_score = 250
        y_best = 300
        y_click = 400
        block_top = y_game_over
        block_bottom = y_click
        block_center = (block_top + block_bottom) / 2
        vertical_offset = (HEIGHT / 2) - block_center

        screen.draw.text(
            "Game Over",
            center=(WIDTH / 2, y_game_over + vertical_offset),
            fontsize=60,
            color="red",
            fontname=GANG_OF_THREE_FONT,
            owidth=2,
            ocolor="black",
        )
        screen.draw.text(
            f"Score: {score}",
            center=(WIDTH / 2, y_score + vertical_offset),
            fontsize=40,
            color="white",
            fontname=GANG_OF_THREE_FONT,
            owidth=2,
            ocolor="black",
        )
        screen.draw.text(
            f"Best: {best_score}",
            center=(WIDTH / 2, y_best + vertical_offset),
            fontsize=40,
            color="white",
            fontname=GANG_OF_THREE_FONT,
            owidth=2,
            ocolor="black",
        )
        screen.draw.text(
            "Click to return to menu",
            center=(WIDTH / 2, y_click + vertical_offset),
            fontsize=30,
            color="white",
            fontname=GANG_OF_THREE_FONT,
            owidth=2,
            ocolor="black",
        )


# Overlays a translucent layer and shows “Paused” with resume/menu buttons.
def draw_paused_screen():
    for splash in splashes:
        splash.actor.draw()
    for fruit in fruits:
        fruit.actor.draw()

    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    screen.surface.blit(overlay, (0, 0))
    screen.draw.text(
        "Paused",
        center=(WIDTH / 2, HEIGHT / 2 - 200),
        fontsize=80,
        color="#E79435",
        fontname=GANG_OF_THREE_FONT,
        owidth=2,
        ocolor="black",
    )
    screen.draw.text(
        f"Score: {score}\nBest: {best_score}",
        center=(WIDTH / 2, HEIGHT / 2 - 80),
        fontsize=50,
        color="#E79435",
        fontname=GANG_OF_THREE_FONT,
        owidth=2,
        ocolor="black",
    )
    resume_rect = Rect((WIDTH / 2 - 380, HEIGHT / 2 + 70), (350, 122))
    menu_rect = Rect((WIDTH / 2, HEIGHT / 2 + 70), (350, 122))
    screen.blit("button_bg", resume_rect.topleft)
    screen.draw.text(
        "Resume Game",
        center=resume_rect.center,
        fontsize=45,
        color="#5A171E",
        fontname=GANG_OF_THREE_FONT,
    )
    screen.blit("button_bg", menu_rect.topleft)
    screen.draw.text(
        "Main Menu",
        center=menu_rect.center,
        fontsize=45,
        color="#5A171E",
        fontname=GANG_OF_THREE_FONT,
    )


# Draws the trailing‐cursor effect with a custom cursor.
def draw_custom_cursor():
    for x, y, life in cursor_trail:
        alpha = int(255 * (life / 20))
        radius = int(10 * (life / 20))
        draw_transparent_circle((x, y), radius, (255, 255, 255), alpha)

    cursor_pos = pygame.mouse.get_pos()
    screen.blit("cursor", (cursor_pos[0] - 10, cursor_pos[1] - 10))


# Central draw dispatcher called by pgzero each frame.
def draw():
    if current_screen == "menu":
        draw_menu_screen()
    else:
        screen.blit("background", (0, 0))

        if current_screen == "game":
            draw_game_screen()
        elif current_screen == "paused":
            draw_paused_screen()
        elif current_screen == "gameover":
            draw_gameover_screen()

    draw_custom_cursor()


# Handles mouse‐move events: builds slice trails and checks for slicing fruits.
def on_mouse_move(pos):
    global score, current_screen, best_score, splashes, cursor_trail, last_trail_position
    if mouse_held:
        if last_trail_position is not None:
            dx = pos[0] - last_trail_position[0]
            dy = pos[1] - last_trail_position[1]
            distance = math.hypot(dx, dy)
            if distance > 5:
                steps = min(int(distance // 5), MAX_TRAIL_STEPS)
                for i in range(1, steps + 1):
                    new_x = last_trail_position[0] + (dx * i / steps)
                    new_y = last_trail_position[1] + (dy * i / steps)
                    cursor_trail.append((new_x, new_y, 20))
            else:
                cursor_trail.append((pos[0], pos[1], 20))
        else:
            cursor_trail.append((pos[0], pos[1], 20))

        last_trail_position = pos
    if current_screen == "game" and mouse_held:
        for f in fruits:
            if not f.getSliced() and f.actor.collidepoint(pos):
                if f.is_bomb:
                    trigger_gameover(True)
                    if score > best_score:
                        best_score = score
                        save(str(best_score))
                else:
                    f.setSliced()
                    score += 1

                    now = pygame.time.get_ticks()
                    rt = (now - f.spawn_time) / 1000.0
                    performance_history.append(True)
                    performance_data.append(
                        {
                            "accuracy": 1.0,
                            "reaction_time": rt,
                            "attempts": session_attempts,
                            "level": (
                                4
                                if selected_difficulty == MODE_AI
                                else list(DIFFICULTIES.keys()).index(
                                    selected_difficulty
                                )
                            ),
                        }
                    )
                if len(splashes) < 5:
                    splashes.append(Splash((f.actor.x, f.actor.y)))
                else:
                    splashes = [Splash((f.actor.x, f.actor.y))]


# Handles mouse‐down click events: UI interaction and game state transitions.
def on_mouse_down(pos):
    global current_screen, best_score, mouse_held, selected_difficulty
    mouse_held = True

    if current_screen == "menu":
        for i, name in enumerate((DIFFICULTIES | {MODE_AI: None})):
            x = 100
            y = ((HEIGHT / 2) - 53) + (i * 60)
            rect = Rect((x - 60, y - 30), (151, 53))
            if rect.collidepoint(pos):
                save(difficulty=name)
                selected_difficulty = name
                return

        start_rect = Rect((WIDTH / 2 - 175, HEIGHT - 200), (350, 122))
        if start_rect.collidepoint(pos):
            current_screen = "game"
            reset_game()
    elif current_screen == "gameover":
        current_screen = "menu"
    elif current_screen == "game":
        pause_button_rect = Rect((80, HEIGHT - 100), (100, 50))
        if pause_button_rect.collidepoint(pos):
            current_screen = "paused"
            return
    elif current_screen == "paused":
        resume_rect = Rect((WIDTH / 2 - 380, HEIGHT / 2 + 70), (350, 122))
        menu_rect = Rect((WIDTH / 2, HEIGHT / 2 + 70), (350, 122))
        if resume_rect.collidepoint(pos):
            current_screen = "game"
            return
        elif menu_rect.collidepoint(pos):
            current_screen = "menu"
            return


# Ends a mouse‐drag/slice action.
def on_mouse_up(pos):
    global mouse_held, last_trail_position
    mouse_held = False
    last_trail_position = None


# Sets game title and icon.
TITLE = "Fruit Ninja Clone"
ICON = "images/icon.png"

# Aligns the game to the center of the screen
os.environ["SDL_VIDEO_CENTERED"] = "1"

# Sets the screen size and checks for overflow with monitor's size
monitor = get_monitors()[0]
WIDTH = min(1920, monitor.width)
HEIGHT = min(800, monitor.height)

# Hides the system cursor
pygame.mouse.set_visible(False)

# Runs the game through pgzero
pgzrun.go()