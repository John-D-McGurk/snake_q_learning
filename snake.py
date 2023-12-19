import pygame
import random
from pygame.math import Vector2
import sys

pygame.init()
cell_size = 40
cell_number = 30
screen = pygame.display.set_mode((cell_number * cell_size, cell_number * cell_size))
clock = pygame.time.Clock()
apple = pygame.image.load('graphics/apple.png').convert_alpha()
game_font = pygame.font.Font('fonts/PlaypenSans-Regular.ttf', 25)


class Snake:
    def __init__(self):
        self.START_POS = [Vector2(5, 10), Vector2(4, 10), Vector2(3, 10)]
        self.body = self.START_POS
        self.direction = Vector2(1, 0)
        self.new_block = False

        self.head_up = pygame.image.load('graphics/head_up.png').convert_alpha()
        self.head_right = pygame.image.load('graphics/head_right.png').convert_alpha()
        self.head_down = pygame.image.load('graphics/head_down.png').convert_alpha()
        self.head_left = pygame.image.load('graphics/head_left.png').convert_alpha()

        self.tail_up = pygame.image.load('graphics/tail_up.png').convert_alpha()
        self.tail_right = pygame.image.load('graphics/tail_right.png').convert_alpha()
        self.tail_down = pygame.image.load('graphics/tail_down.png').convert_alpha()
        self.tail_left = pygame.image.load('graphics/tail_left.png').convert_alpha()

        self.body_vertical = pygame.image.load('graphics/body_vertical.png').convert_alpha()
        self.body_horizontal = pygame.image.load('graphics/body_horizontal.png').convert_alpha()

        self.body_tr = pygame.image.load('graphics/body_topright.png').convert_alpha()
        self.body_tl = pygame.image.load('graphics/body_topleft.png').convert_alpha()
        self.body_bl = pygame.image.load('graphics/body_bottomleft.png').convert_alpha()
        self.body_br = pygame.image.load('graphics/body_bottomright.png').convert_alpha()

        

    def draw_snake(self):
        self.update_head_graphics()
        self.update_tail_graphics()

        for index, block in enumerate(self.body):
            x_pos = int(block.x * cell_size)
            y_pos = int(block.y * cell_size)
            block_rect =pygame.Rect(x_pos, y_pos, cell_size, cell_size)

            if index == 0:
                screen.blit(self.head,block_rect)
            elif index == len(self.body) - 1:
                screen.blit(self.tail, block_rect)
            else:
                prev_block = self.body[index + 1] - block
                next_block = self.body[index - 1] - block
                if prev_block.x == next_block.x:
                    screen.blit(self.body_vertical, block_rect)
                elif prev_block.y == next_block.y:
                    screen.blit(self.body_horizontal, block_rect)
                else:
                    if prev_block.x == -1 and next_block.y == 1 or prev_block.y == 1 and next_block.x == -1:
                        screen.blit(self.body_bl, block_rect)
                    elif prev_block.x == -1 and next_block.y == -1 or prev_block.y == -1 and next_block.x == -1:
                        screen.blit(self.body_tl, block_rect)
                    elif prev_block.x == 1 and next_block.y == -1 or prev_block.y == -1 and next_block.x == 1:
                        screen.blit(self.body_tr, block_rect)
                    else:
                        screen.blit(self.body_br, block_rect)

    def update_head_graphics(self):
        head_direction = self.body[1] - self.body[0]
        if head_direction == Vector2(-1, 0):
            self.head = self.head_right
        elif head_direction == Vector2(0, -1):
            self.head = self.head_down
        elif head_direction == Vector2(1, 0):
            self.head = self.head_left
        else:
            self.head = self.head_up

    def update_tail_graphics(self):
        tail_direction = self.body[-2] - self.body[-1]
        if tail_direction == Vector2(-1, 0):
            self.tail = self.tail_right
        elif tail_direction == Vector2(0, -1):
            self.tail = self.tail_down
        elif tail_direction == Vector2(1, 0):
            self.tail = self.tail_left
        else:
            self.tail = self.tail_up


    def move_snake(self):
        if self.new_block:
            body_copy = self.body[:]
            self.new_block = False
        else:
            body_copy = self.body[:-1]

        body_copy.insert(0, body_copy[0] + self.direction)
        self.body = body_copy
    
    def add_block(self):
        self.new_block = True

    def reset(self):
        self.body = self.START_POS
        self.direction = Vector2(1, 0)

class Fruit:
    def __init__(self):
        self.randomize()
        self.fruit = apple

    def draw_fruit(self):
        x_pos = int(self.pos.x * cell_size)
        y_pos = int(self.pos.y * cell_size)

        fruit_rect = pygame.Rect(x_pos, y_pos, cell_size, cell_size)
        screen.blit(self.fruit, fruit_rect)
    
    def randomize(self):
        self.x = random.randint(0, cell_number - 1)
        self.y = random.randint(0, cell_number - 1)
        self.pos = Vector2(self.x, self.y)

class Main:
    def __init__(self):
        self.snake = Snake()
        self.fruit = Fruit()

    def update(self,randomize_fruit=True):
        self.snake.move_snake()
        self.check_collision(randomize_fruit)
        self.check_fail()


    def draw_elements(self):
        self.draw_grass()
        self.snake.draw_snake()
        self.fruit.draw_fruit()
        self.draw_score()

    def check_collision(self, randomize_fruit=True):
        if self.fruit.pos == self.snake.body[0]:
            self.snake.add_block()
            if randomize_fruit:
                self.fruit.randomize()
                while self.fruit.pos in self.snake.body:
                    self.fruit.randomize()

    def check_fail(self):
        if not 0 <= self.snake.body[0].x < cell_number or not 0 <= self.snake.body[0].y < cell_number:
            self.game_over()
    
        if self.snake.body[0] in self.snake.body[1:]:
            self.game_over()

    def game_over(self):
        self.snake.reset()
        self.fruit.randomize()

    def draw_grass(self):
        grass_color = (167, 209, 61)
        for col in range(cell_number):
            for row in range(cell_number):
                if abs(col - row) % 2 == 0:
                    grass_rect = (pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size))
                    pygame.draw.rect(screen, grass_color, grass_rect)
    
    def draw_score(self):
        score_text = str(len(self.snake.body) - 3)
        score_surface = game_font.render(score_text, True, (10, 10, 10))
        score_x = int(cell_size * cell_number - 60)
        score_y = int(cell_size * cell_number - 60)
        score_rect = score_surface.get_rect(center=(score_x, score_y))
        apple_rect = apple.get_rect(midright=(score_rect.midleft))
        bg_rect = pygame.Rect(apple_rect.left, apple_rect.top, apple_rect.width + score_rect.width + 6 , apple_rect.height)

        pygame.draw.rect(screen, (10, 10, 10), bg_rect,2)
        screen.blit(score_surface, score_rect)
        screen.blit(apple, apple_rect)
    

if __name__ == "__main__":
    SCREEN_UPDATE = pygame.USEREVENT
    pygame.time.set_timer(SCREEN_UPDATE, 150)

    main_game = Main()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == SCREEN_UPDATE:
                main_game.update()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    if main_game.snake.direction.y != 1:
                        main_game.snake.direction = Vector2(0, -1)
                if event.key == pygame.K_RIGHT:
                    if main_game.snake.direction.x != -1:
                        main_game.snake.direction = Vector2(1, 0)
                if event.key == pygame.K_DOWN:
                    if main_game.snake.direction.y != -1:
                        main_game.snake.direction = Vector2(0, 1)
                if event.key == pygame.K_LEFT:
                    if main_game.snake.direction.x != 1:
                        main_game.snake.direction = Vector2(-1, 0)
        

        screen.fill((175,215,70))
        main_game.draw_elements()
        pygame.display.update()
        clock.tick(60)