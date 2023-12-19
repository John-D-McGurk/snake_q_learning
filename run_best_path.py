import snake as game
import numpy as np
import pygame 
from pygame.math import Vector2
import random as rnd
import sys

####################
#   Constants 
####################
# Directions
NO_MOVEMENT = Vector2(0, 0)
UP = Vector2(0, -1)
RIGHT = Vector2(1, 0)
DOWN = Vector2(0, 1)
LEFT = Vector2(-1, 0)

# Rewards
REWARD_COLLISION = -10

# Game constants
CELL_SIZE = 40
CELL_NUMBER = 20
SCREEN_WIDTH = CELL_NUMBER * CELL_SIZE
SCREEN_HEIGHT = CELL_NUMBER * CELL_SIZE


PATH = np.load("paths/85_points.npy").astype(int)


if __name__ == "__main__":


####################
#   Initialisation 
####################
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    clock = pygame.time.Clock()
    apple = pygame.image.load('graphics/apple.png').convert_alpha()
    game_font = pygame.font.Font('fonts/PlaypenSans-Regular.ttf', 25)

    SCREEN_UPDATE = pygame.USEREVENT
    pygame.time.set_timer(SCREEN_UPDATE, 50)
    time = 50
    pygame.display.set_caption('Snake - Run AI')
    pygame.display.set_icon(apple)

    main_game = game.Main()

def quit_game():
    """ Quits the game"""
    pygame.quit()
    sys.exit()

####################
#   Game loop 
####################
move = 0
while True:
    main_game.fruit.pos = Vector2(tuple(PATH[1][move]))
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            quit_game()

        if event.type == SCREEN_UPDATE:
            main_game.snake.direction = Vector2(tuple(PATH[0][move]))
            
            main_game.update(False)
            move += 1

        if event.type == pygame.KEYDOWN:

            # Set plus and minus keys to increase or decrease play speed, respectively
            if event.key == pygame.K_MINUS:
                time += 10
                pygame.time.set_timer(SCREEN_UPDATE, time)
            elif event.key == pygame.K_PLUS:
                
                if time > 10:
                    time -=10
                else:
                    time = 1

                pygame.time.set_timer(SCREEN_UPDATE, time)

    # Draw elements and update display
    screen.fill((175,215,70))
    main_game.draw_elements()

    pygame.display.update()
    clock.tick(60)