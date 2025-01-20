import pygame
from enum import Enum
from typing import Optional, Union, List
import numpy as np
import torch

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong Game")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Paddle dimensions
PADDLE_WIDTH, PADDLE_HEIGHT = 10, 100

# Ball dimensions
BALL_SIZE = 20

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
        self.speed = 7
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
        if self.rect.centery < ball.rect.centery:
            self.rect.y += self.speed
            self.direction = Direction.UP
        if self.rect.centery > ball.rect.centery:
            self.rect.y -= self.speed
            self.direction = Direction.DOWN

    def draw(self, screen):
        pygame.draw.rect(screen, WHITE, self.rect)


class Ball:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, BALL_SIZE, BALL_SIZE)
        self.speed_x = 3
        self.speed_y = 3

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
        pygame.draw.aaline(self.screen, WHITE, (WIDTH // 2, score_text.get_width()*2), (WIDTH // 2, HEIGHT))
        self.screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, 10))
        pygame.display.flip()

    def get_snapshot(self) -> np.ndarray:
        return pygame.surfarray.array2d(self.screen)

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

        # if self.ball.check_collision(self.player):
        #     self.score += 1

        if self.ball.check_collision(self.opponent) or self.ball.check_collision(self.player):
            self.ball.speed_x += np.random.random()
            self.ball.speed_y += np.random.random()

        game_over = False
        if self.ball.rect.left <= 0:
            self.score += 1
            self.ball_reset()
            # game_over = True
        if self.ball.rect.left >= WIDTH:
            self.score -= 1
            self.ball_reset()
            # game_over = True

        if self.score <= -4:
            game_over = True

        self.draw()
        self.clock.tick(60)
        
        return game_over, self.score

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