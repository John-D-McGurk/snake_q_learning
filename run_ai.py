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
REWARD_FRUIT = 1
REWARD_COLLISION = -1

# Game constants
CELL_SIZE = 40
CELL_NUMBER = 20
SCREEN_WIDTH = CELL_NUMBER * CELL_SIZE
SCREEN_HEIGHT = CELL_NUMBER * CELL_SIZE + 100


####################
#   Agent 
####################
class Robot:
    def __init__(self):
        """ Initialise robot object, with reward matrix and empty Q matrix"""
        self.best_score = 0
        self.current_path = np.zeros(0)
        self.best_path = np.zeros(0)
        self.scores = np.zeros(0, float)
        self.q_matrix = np.load('trained_q_matrices/trained_array_2000.npy')
        self.current_state = self.get_state()
    
    
    def get_state(self, next_direction=NO_MOVEMENT):
        """ Return state as a NumPy array.
        # Inputs:
        next_direction  Vector2 defining new movement direction for snake (default (0, 0), to return current state)

        # Returns:
        state           NumPy array containing 12 dimensions defining the current state 
                        and 1 dimension defining next direction
        
        # Notes:

        state[0]   = snake is moving up
        state[1]   = snake is moving right
        state[2]   = snake is moving down
        state[3]   = snake is moving left
        state[4]   = food is up
        state[5]   = food is right
        state[6]   = food is down
        state[7]   = food is left
        state[8]   = danger up
        state[9]   = danger right
        state[10]  = danger down
        state[11]  = danger left
        """
        state = np.zeros(12, dtype=int)
        if next_direction == NO_MOVEMENT:
            direction = main_game.snake.direction
        else:
            direction = next_direction


        if direction == UP:
            state[0] = 1
        elif direction == RIGHT:
            state[1] = 1
        elif direction == DOWN:
            state[2] = 1
        else:
            state[3] = 1
        
        food = main_game.fruit.pos
        snake_head = main_game.snake.body[0] + next_direction
        if food.y < snake_head[1]:
            state[4] = 1
        elif food.y > snake_head[1]:
            state[6] = 1
        if food.x > snake_head[0]:
            state[5] = 1
        elif food.x < snake_head[0]:
            state[7] = 1

        body = main_game.snake.body[:-1]
        if snake_head + UP in body or (snake_head + UP).y >= CELL_NUMBER:
            state [8] = 1
        if snake_head + RIGHT in body or (snake_head + RIGHT).x >= CELL_NUMBER:
            state[9] = 1
        if snake_head + DOWN in body or (snake_head + DOWN).y >= CELL_NUMBER:
            state[10] = 1
        if snake_head + LEFT in body or (snake_head + LEFT).x < 0:
            state[11] = 1

        return state
    
    def vectorize_direction(self, direction):
        """ Return vector version of integer directions.
        
        # Inputs:
        direction           Integer defining movement direction
        
        # Returns:
        direction_vector    Vector2 defining movement direction
        """
        if direction == 0:
            return UP
        elif direction == 1:
            return RIGHT
        elif direction == 2:
            return DOWN
        else:
            return LEFT

    def detect_collision(self, next_direction):
        """ Return the reward from moving into the next state.
        
        # Inputs:
        next_direction: Vector2 defining next movement direction chosen

        # Returns:
        reward:         Reward for moving to the next state
        
        # Notes:
        If next_state is equal to current_state, this means the robot attempted to move
        outside of the map. In this scenario the function returns the value associated with EXIT_MAP
        """
        next_pos = main_game.snake.body[0] + next_direction
        if next_pos == main_game.fruit.pos:
            return REWARD_FRUIT
        elif not 0 <= next_pos.x < CELL_NUMBER or not 0 <= next_pos.y < CELL_NUMBER:
            return REWARD_COLLISION
        elif next_pos in main_game.snake.body[:-1]:
            return REWARD_COLLISION
        
    def get_next_state_monte_carlo(self, current_state):
        """ Return a next state randomly."""
        # Choose possible state that doesn't send snake back into itself    
        if current_state[0]:
            possible_directions = [0,1,3]
        elif current_state[1]:
            possible_directions = [0,1,2]
        elif current_state[2]:
            possible_directions = [1,2,3]
        else:
            possible_directions = [0,2,3]

        selection = rnd.randint(0, (len(possible_directions) - 1))
        direction = possible_directions[selection]
        return direction

    
        
    def get_next_state_greedy(self):
        """ Return next state based on best value accoding to Q-martix.
        
        # Inputs:
        current_state   NumPy array containing 12 dimensions defining the current state 
                        and 1 dimension defining next direction
                        
        # Returns:
        direction       Integer defining the chosen next direction
        """
        best_choice = float('-inf')
        q_scores = []

        for i in range(4):
            current_q_score = self.q_matrix[tuple(self.current_state)][i]
            if current_q_score > best_choice:
                best_choice = current_q_score
                direction = i
            
            q_scores.append(current_q_score)

        # If 2 or more directions yield the same reward, select one randomly with monte carlo        
        if q_scores.count(max(q_scores)) > 1:
            optimum_directions = [index for (index, item) in enumerate(q_scores) if item == max(q_scores)]
            return self.get_next_state_monte_carlo(optimum_directions)
        # Else return reward of optimum direction
        else:
            return direction
    
    def store_path(self, next_direction_vector):
        """ Assigns movements and fruit locations to self.current_path array
        
        # Inputs:
        next_direction_vector:  Vector2 selecting next movement
        """
        if not len(self.current_path):
            self.current_path = np.array([[next_direction_vector], [main_game.fruit.pos]])
        else:
            new_column = np.array([[next_direction_vector], [main_game.fruit.pos]])
            self.current_path = np.append(self.current_path, new_column, 1)


    def store_game_info(self):
        score = len(main_game.snake.body) - 3

        if score > self.best_score:
            self.best_score = score
            self.best_path = self.current_path
        
        self.current_path = np.zeros(0, int)
        

    def greedy_path(self, max_epochs=float('inf')):
        """ Run with a greedy policy with a trained Q matrix and return the best reward and score with greedy-path()
        # Inputs:
        max_epochs:   Number of epochs (simulations) to run
        """

        self.current_state = self.get_state()
        
        next_direction = self.get_next_state_greedy()
        next_direction_vector = self.vectorize_direction(next_direction)

        current_reward = self.detect_collision(next_direction_vector)

        self.store_path(next_direction_vector)


        main_game.snake.direction = next_direction_vector

        if current_reward == REWARD_COLLISION:
            self.store_game_info()
            

            



####################
#   Game functions 
####################
def draw_values():
    """ Draws best score to game screen."""
    values_y = int(CELL_SIZE * CELL_NUMBER + 50)

    best_score_text = f"Best: {robot.best_score}"
    best_score_x = int(CELL_SIZE * CELL_NUMBER - 200)
    best_score_surface = game_font.render(best_score_text, True, (10, 10, 10))
    best_score_rect = best_score_surface.get_rect(midleft=(best_score_x, values_y))
    
    screen.blit(best_score_surface, best_score_rect)

def quit_game():
    """ Quits the game, saves the best path, prints best score and plots the score over epochs"""
    np.save(f"paths/{robot.best_score}_points",robot.best_path)
    print(f"Best score: {robot.best_score}")
    pygame.quit()
    sys.exit()


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
    pygame.time.set_timer(SCREEN_UPDATE, 1)
    time = 1
    pygame.display.set_caption('Snake - Run AI')
    pygame.display.set_icon(apple)

    main_game = game.Main()

    robot = Robot()


####################
#   Game loop 
####################
    while True:
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                quit_game()

            if event.type == SCREEN_UPDATE:
                robot.greedy_path()
                main_game.update()

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
        draw_values()
        pygame.display.update()
        clock.tick(60)