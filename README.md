# AI-Adaptive Fruit Ninja Clone (LAS205)

A Fruit Ninja–style arcade game built in Python (Pygame Zero) for my LAS205 university project. The core gameplay focuses on responsive slicing, clean UI flow (menu/pause/game-over), and an optional AI-assisted difficulty mode that adapts to the player using a lightweight TensorFlow/Keras model trained on recent performance (accuracy, reaction time, attempts).

## Features
- Classic fruit-slicing gameplay with scoring and “miss” tracking (game ends after 3 missed fruits)
- Bomb hazard (instant game over if sliced)
- Difficulty presets: Eaasy / Normal / Hard / Impossible
- AI Detect mode:
  - Logs performance to `performance_data.csv`
  - Trains/updates a simple neural network and predicts an appropriate difficulty level
  - Adjusts spawn pacing based on recent performance
- Menu + pause screen + animated game-over sequence
- Custom cursor + trailing slice effect
- Best score saving/loading