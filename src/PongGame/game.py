"""Pure-numpy Pong. No pygame, no display — fast for vectorized RL rollouts."""
from enum import Enum
from typing import Optional, Tuple
import numpy as np

WIDTH, HEIGHT = 140, 100
PADDLE_WIDTH, PADDLE_HEIGHT = 3, 14
BALL_SIZE = 5
MAX_SCORE = 4
MAX_BALL_SPEED = 3.0


class Direction(Enum):
    DOWN = 0
    UP = 1
    STOP = 2


class Rect:
    __slots__ = ("x", "y", "w", "h")

    def __init__(self, x, y, w, h):
        self.x = float(x)
        self.y = float(y)
        self.w = w
        self.h = h

    @property
    def top(self): return self.y
    @property
    def bottom(self): return self.y + self.h
    @property
    def left(self): return self.x
    @property
    def right(self): return self.x + self.w
    @property
    def centery(self): return self.y + self.h / 2

    def colliderect(self, other: "Rect") -> bool:
        return (self.left < other.right and self.right > other.left
                and self.top < other.bottom and self.bottom > other.top)


class Paddle:
    def __init__(self, x, y, rng: np.random.Generator):
        self.rect = Rect(x, y, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.speed = 4
        self.rng = rng

    def move(self, step: Direction):
        if step == Direction.UP and self.rect.top > 0:
            self.rect.y -= self.speed
        elif step == Direction.DOWN and self.rect.bottom < HEIGHT:
            self.rect.y += self.speed

    def ai_move(self, ball: "Ball"):
        if self.rng.random() < 0.4:
            return
        target_y = ball.rect.centery
        if self.rect.centery < target_y:
            self.rect.y += self.speed * 0.5
        elif self.rect.centery > target_y:
            self.rect.y -= self.speed * 0.5


class Ball:
    def __init__(self, x, y, rng: np.random.Generator):
        self.rect = Rect(x, y, BALL_SIZE, BALL_SIZE)
        self.speed_x = 1.0 if rng.integers(0, 2) == 0 else -1.0
        self.speed_y = 1.0 if rng.integers(0, 2) == 0 else -1.0

    def move(self):
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y
        if self.rect.top <= 0 or self.rect.bottom >= HEIGHT:
            self.speed_y *= -1

    def check_collision(self, paddle: Paddle) -> bool:
        if not self.rect.colliderect(paddle.rect):
            return False
        ball_cx = self.rect.x + self.rect.w / 2
        paddle_cx = paddle.rect.x + paddle.rect.w / 2
        if ball_cx < paddle_cx and self.speed_x > 0:
            self.rect.x = paddle.rect.left - self.rect.w
            self.speed_x = -abs(self.speed_x)
            return True
        if ball_cx > paddle_cx and self.speed_x < 0:
            self.rect.x = paddle.rect.right
            self.speed_x = abs(self.speed_x)
            return True
        return False


class PongGame:
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.score = 0
        self.player = Paddle(WIDTH - 20, HEIGHT // 2 - PADDLE_HEIGHT // 2, self.rng)
        self.opponent = Paddle(10, HEIGHT // 2 - PADDLE_HEIGHT // 2, self.rng)
        self._ball_reset()

    def _ball_reset(self):
        self.ball = Ball(WIDTH // 2 - BALL_SIZE // 2,
                         HEIGHT // 2 - BALL_SIZE // 2, self.rng)

    def get_state_vector(self) -> np.ndarray:
        return np.array([
            self.ball.rect.x / WIDTH,
            self.ball.rect.y / HEIGHT,
            self.ball.speed_x / 5.0,
            self.ball.speed_y / 5.0,
            self.player.rect.y / HEIGHT,
            self.opponent.rect.y / HEIGHT,
        ], dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        if action == 0:
            step = Direction.DOWN
        elif action == 1:
            step = Direction.UP
        else:
            step = Direction.STOP

        self.player.move(step)
        self.opponent.ai_move(self.ball)
        self.ball.move()

        reward = 0.0
        ball_y = self.ball.rect.centery
        paddle_center = self.player.rect.centery
        distance_to_ball = abs(ball_y - paddle_center)

        if self.ball.speed_x > 0:
            max_distance = HEIGHT / 2
            reward += 0.05 * (1 - min(distance_to_ball / max_distance, 1.0))

        if self.ball.check_collision(self.player):
            reward += 2.0

        if self.ball.rect.right >= self.player.rect.left and self.ball.speed_x > 0:
            if distance_to_ball > PADDLE_HEIGHT:
                reward -= 0.5

        if self.ball.check_collision(self.opponent):
            self.ball.speed_x += self.rng.random()
            self.ball.speed_y += self.rng.random()
            self.ball.speed_x = float(np.clip(self.ball.speed_x, -MAX_BALL_SPEED, MAX_BALL_SPEED))
            self.ball.speed_y = float(np.clip(self.ball.speed_y, -MAX_BALL_SPEED, MAX_BALL_SPEED))

        done = False
        if self.ball.rect.left <= 0:
            self.score += 1
            reward += 5.0
            self._ball_reset()
        if self.ball.rect.left >= WIDTH:
            self.score -= 1
            reward -= 5.0
            self._ball_reset()

        if abs(self.score) >= MAX_SCORE:
            done = True

        return self.get_state_vector(), reward, done

    def render(self) -> np.ndarray:
        return render_frame(self.ball.rect.x, self.ball.rect.y,
                            self.player.rect.y, self.opponent.rect.y)


def render_frame(ball_x: float, ball_y: float,
                 player_y: float, opponent_y: float) -> np.ndarray:
    """Render a Pong frame as an (HEIGHT, WIDTH, 3) uint8 RGB array."""
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    img[:, WIDTH // 2] = 80  # center line
    px, py = WIDTH - 20, int(player_y)
    img[max(0, py):min(HEIGHT, py + PADDLE_HEIGHT),
        max(0, px):min(WIDTH, px + PADDLE_WIDTH)] = 255
    ox, oy = 10, int(opponent_y)
    img[max(0, oy):min(HEIGHT, oy + PADDLE_HEIGHT),
        max(0, ox):min(WIDTH, ox + PADDLE_WIDTH)] = 255
    bx, by = int(ball_x), int(ball_y)
    img[max(0, by):min(HEIGHT, by + BALL_SIZE),
        max(0, bx):min(WIDTH, bx + BALL_SIZE)] = 255
    return img


def render_from_state(state: np.ndarray) -> np.ndarray:
    """Render frame from a normalized state vector (see get_state_vector)."""
    return render_frame(state[0] * WIDTH, state[1] * HEIGHT,
                        state[4] * HEIGHT, state[5] * HEIGHT)
