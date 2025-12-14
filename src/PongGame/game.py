import pygame
from enum import Enum
from typing import Optional, Union, List
import numpy as np
import torch
import random

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 140, 100
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong Game")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Paddle dimensions
PADDLE_WIDTH, PADDLE_HEIGHT = 3, 14

# Ball dimensions
BALL_SIZE = 5

# Paddle positions
player_x, player_y = WIDTH - 20, HEIGHT // 2 - PADDLE_HEIGHT // 2
opponent_x, opponent_y = 10, HEIGHT // 2 - PADDLE_HEIGHT // 2


class Direction(Enum):
    DOWN = 0
    UP = 1
    STOP = 2


class Paddle:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.speed = 4
        self.direction = Direction.STOP

    def move(self, step: Direction):
        # print(step)
        if step == Direction.UP and self.rect.top > 0:
            self.rect.y -= self.speed
        elif step == Direction.DOWN and self.rect.bottom < HEIGHT:
            self.rect.y += self.speed
        elif step == Direction.STOP:
            self.rect.y += 0

    def ai_move(self, ball):
        # Add randomness and delay to make AI less perfect
        # Making AI weaker for initial learning
        if np.random.random() < 0.4:  # 40% chance to make a mistake (was 10%)
            return  # Skip movement occasionally
        
        # Add some prediction error
        target_y = ball.rect.centery
        
        # Slower reaction speed - make it beatable
        if self.rect.centery < target_y:
            self.rect.y += self.speed * 0.5  # Reduced speed
            self.direction = Direction.UP
        if self.rect.centery > target_y:
            self.rect.y -= self.speed * 0.5  # Reduced speed
            self.direction = Direction.DOWN

    def draw(self, screen):
        pygame.draw.rect(screen, WHITE, self.rect)


class Ball:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, BALL_SIZE, BALL_SIZE)
        self.speed_x, self.speed_y = self._init_speed()

    def _init_speed(self):
        # Randomize both X and Y directions to avoid training bias
        x_speed = 1 if random.randint(0, 1) == 0 else -1
        y_speed = 1 if random.randint(0, 1) == 0 else -1
        return x_speed, y_speed

    def move(self):
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y

        if self.rect.top <= 0 or self.rect.bottom >= HEIGHT:
            self.speed_y *= -1

    def check_collision(self, paddle):
        if self.rect.colliderect(paddle.rect):
            self.speed_x *= -1
            return True
        return False

    def draw(self, screen):
        pygame.draw.ellipse(screen, WHITE, self.rect)


class PongGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Pong Game")
        self.reset()

    def reset(self):
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 74)
        self.score = 0
        self.player = Paddle(WIDTH - 20, HEIGHT // 2 - PADDLE_HEIGHT // 2)
        self.opponent = Paddle(10, HEIGHT // 2 - PADDLE_HEIGHT // 2)
        self.ball = Ball(WIDTH // 2 - BALL_SIZE // 2, HEIGHT // 2 - BALL_SIZE // 2)

    def ball_reset(self):
        self.ball = Ball(WIDTH // 2 - BALL_SIZE // 2, HEIGHT // 2 - BALL_SIZE // 2)

    def draw(self):
        self.screen.fill(BLACK)
        self.player.draw(self.screen)
        self.opponent.draw(self.screen)
        self.ball.draw(self.screen)
        score_text = self.font.render(str(self.score), True, WHITE)
        pygame.draw.aaline(self.screen, WHITE, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT))
        pygame.display.flip()

    def get_snapshot(self) -> np.ndarray:
        # Get RGB array (width x height x 3)
        rgb_array = pygame.surfarray.array3d(self.screen)
        # Convert to grayscale using standard luminosity method
        # Formula: 0.299*R + 0.587*G + 0.114*B
        grayscale = np.dot(rgb_array[..., :3], [0.299, 0.587, 0.114])
        return grayscale.astype(np.uint8)  # Returns (140, 100) shape

    def get_state_vector(self) -> np.ndarray:
        """Get structured state with positions and velocities (more informative than pixels)"""
        # Normalize all values to [0, 1] range
        return np.array([
            self.ball.rect.x / WIDTH,           # Ball X position
            self.ball.rect.y / HEIGHT,          # Ball Y position
            self.ball.speed_x / 5.0,            # Ball X velocity (normalized, can be negative)
            self.ball.speed_y / 5.0,            # Ball Y velocity (normalized, can be negative)
            self.player.rect.y / HEIGHT,        # Player paddle Y position
            self.opponent.rect.y / HEIGHT,      # Opponent paddle Y position
        ], dtype=np.float32)

    def get_step_from_event(self, event: pygame.event.Event) -> Direction:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                return Direction.UP
            elif event.key == pygame.K_DOWN:
                return Direction.DOWN
        return Direction.STOP

    def get_step_from_tensor(self, value: torch.Tensor) -> Direction:
        if value in [0, 1]:
            return Direction(value.item())
        return Direction.STOP
    
    def get_step_from_int(self, value: int) -> Direction:
        if value in [0, 1]:
            return Direction(value)
        return Direction.STOP

    def _move(self, step: Direction):
        self.player.move(step)
        self.player.direction = step
        self.opponent.ai_move(self.ball)

    def play_step(self, value: Optional[Union[int, List[int]]] = None):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            step = self.get_step_from_event(event)
            self._move(step)

        if value is not None:
            if isinstance(value, int) or isinstance(value, np.int64):
                self.player.direction = self.get_step_from_int(value)
            else:
                self.player.direction = self.get_step_from_tensor(value)
        self._move(self.player.direction)
        self.ball.move()
        
        reward = 0
        
        # Dense reward shaping for better learning
        ball_y = self.ball.rect.centery
        paddle_center = self.player.rect.centery
        distance_to_ball = abs(ball_y - paddle_center)
        
        # Only give alignment reward when ball is approaching player
        # This prevents exploitation by just staying in one position
        if self.ball.speed_x > 0:  # Ball moving toward player
            max_distance = HEIGHT / 2
            alignment_reward = 0.05 * (1 - min(distance_to_ball / max_distance, 1.0))
            reward += alignment_reward
        
        # Strong reward for hitting the ball
        if self.ball.check_collision(self.player):
            reward += 2.0  # Significant reward for successful hit
            
        # Penalty if ball passes paddle (missed hit)
        if self.ball.rect.right >= self.player.rect.left and self.ball.speed_x > 0:
            # Ball is at paddle x-position but moving toward player
            if distance_to_ball > PADDLE_HEIGHT:
                reward -= 0.5  # Penalty for being out of position

        if self.ball.check_collision(self.opponent):
            self.ball.speed_x += np.random.random()
            self.ball.speed_y += np.random.random()

        game_over = False
        if self.ball.rect.left <= 0:
            self.score += 1
            reward += 5.0  # Larger reward for scoring
            self.ball_reset()
        if self.ball.rect.left >= WIDTH:
            self.score -= 1
            reward -= 5.0  # Larger penalty for getting scored on
            self.ball_reset()

        if abs(self.score) >= 4:
            game_over = True

        self.draw()
        self.clock.tick(60)
        
        return game_over, reward

    def draw_game_over(self):
        font = pygame.font.Font(None, self.block_size)
        text = font.render('Game Over - Press Enter to Play Again', True, WHITE)
        text_rect = text.get_rect(center=(WIDTH//2, HEIGHT/2))
        self.screen.blit(text, text_rect)
        pygame.display.flip()
    
    def draw_victory(self):
        font = pygame.font.Font(None, self.block_size)
        text = font.render('You win - Press Enter to Play Again', True, WHITE)
        text_rect = text.get_rect(center=(WIDTH//2, HEIGHT/2))
        self.screen.blit(text, text_rect)
        pygame.display.flip()

if __name__ == "__main__":
    game = PongGame()
    
    while True:
        game_over, score = game.play_step()
        
        if game_over:
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                        game.reset()
                        waiting = False
    
    print(f"Final Score: {score}")
    pygame.quit()