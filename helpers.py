import os

BEST_SCORE_PATH = "./score.data"
DIFFICULTY_PATH = "./difficulty.data"


def load():
    score = 0
    difficulty = "easy"
    if not os.path.isfile(BEST_SCORE_PATH):
        return score, difficulty
    try:
        with open(BEST_SCORE_PATH, "r") as f:
            score = f.read()
    except Exception:
        return score, difficulty

    if not os.path.isfile(DIFFICULTY_PATH):
        return score, difficulty
    try:
        with open(DIFFICULTY_PATH, "r") as f:
            difficulty = f.read()
    except Exception:
        return score, difficulty
    return score, difficulty


def save(score: str | None = None, difficulty: str | None = None):
    if score:
        try:
            with open(BEST_SCORE_PATH, "w+") as f:
                f.write(score)
        except Exception as e:
            print("Error saving score:", e)
    if difficulty:
        try:
            with open(DIFFICULTY_PATH, "w+") as f:
                f.write(difficulty)
        except Exception as e:
            print("Error saving score:", e)
