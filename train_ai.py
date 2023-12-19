import snake as game
import matplotlib.pyplot as plt
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
REWARD_FRUIT = 50
REWARD_COLLISION = -10
REWARD_MOVE = 0

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
        self.alpha = 0.3
        self.gamma = 0.2
        self.epsilon = 0.2
        
        self.epochs = 0
        self.best_score = 0
        self.current_path = np.zeros(2)
        self.best_path = np.zeros(0)
        self.scores = np.zeros(0, float)

        self.q_matrix = np.zeros((2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4))
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

    def get_reward(self, next_direction):
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
        else:
            return REWARD_MOVE
        
    def get_next_state_monte_carlo(self):
        """ Return a next state randomly."""
        # Choose possible state that doesn't send snake back into itself    
        if self.current_state[0]:
            possible_directions = [0,1,3]
        elif self.current_state[1]:
            possible_directions = [0,1,2]
        elif self.current_state[2]:
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
            return self.get_next_state_monte_carlo()
        # Else return reward of optimum direction
        else:
            return direction

    def get_next_state(self):
        """ Uses epsilon greedy policy to select next state.
        
        # Returns:
        next_direction:     Integer, 0-3 selecting direction for next agant state
        """
        generate_next_state_method = rnd.randint(1,100) / 100

        if generate_next_state_method < self.epsilon:
            next_direction = self.get_next_state_monte_carlo()
        else:
            next_direction = self.get_next_state_greedy()
        return next_direction

    def update_q_matrix(self, next_direction, next_state, current_reward):
        
        """ Fills in the Q-matrix using the Bellman Function
        
        # Inputs:
        next_direction      Int, chosen next direction of movement
        next_state          NumPy array containing next state
        current_reward      Int, reward for next move
        """ 
        self.q_matrix[tuple(self.current_state)][next_direction] =\
            (1 - self.alpha)\
            * self.q_matrix[tuple(self.current_state)][next_direction]\
            + self.alpha * (current_reward + self.gamma\
            * max(self.q_matrix[tuple(next_state)]))
            
    def store_path(self, next_direction_vector):
        """ Assigns movements and fruit locations to self.current_path array
        
        # Inputs:
        next_direction_vector:  Vector2 selecting next movement
        """
        if not len(self.current_path):
            self.current_path = np.array([[main_game.snake.direction], [main_game.fruit.pos]])
        else:
            self.current_path = np.append(self.current_path, [[next_direction_vector], [main_game.fruit.pos]])

    def store_game_info(self, print_info=False):
        score = len(main_game.snake.body) - 3
        self.scores = np.append(self.scores, score)
        if print_info:
            print("####################")
            print(f"Epoch: {self.epochs}")
            print(f"Epsilon: {self.epsilon}")
            print(f"Alpha: {self.alpha}")
            print(f"Score: {score}")

        if score > self.best_score:
            self.best_score = score
            self.best_path = self.current_path
        
        self.current_path = np.zeros(0)
        self.decay_function()
        
    def plot_graph(self):
        """ Plots the score and average running score against number of epochs"""
        average_window = round(self.epochs / 10)
        running_average = np.zeros(average_window)
        if len(self.scores > average_window):
            for i in range(average_window, len(self.scores)):
                running_average = np.append(running_average, np.mean(self.scores[i - average_window:i]))

        plt.plot(self.scores)
        plt.plot(running_average)        
        plt.ylabel('score')
        plt.xlabel('number of games')
        plt.show()

    def decay_function(self):
        """ Alpha and epsilon decay functions"""
        if self.alpha > 0.01:
            self.alpha -= 0.0003
        if self.epsilon > 0.0003:
            self.epsilon -= 0.0002

    def q_learning(self, max_epochs=float('inf')):
        """ Run q-learning using epsilon-greedy policy and return the best reward and score with greedy-path()
        # Inputs:
        max_epochs:   Number of epochs (simulations) to run

        # Notes:
        When max_epochs is reached the program will save the trained Q-matrix to an external file
        """

        self.current_state = self.get_state()
        
        next_direction = self.get_next_state()
        next_direction_vector = self.vectorize_direction(next_direction)

        current_reward = self.get_reward(next_direction_vector)

        next_state = self.get_state(next_direction_vector)

        self.store_path(next_direction_vector)

        self.update_q_matrix(next_direction, next_state, current_reward)

        main_game.snake.direction = next_direction_vector

        if current_reward == REWARD_COLLISION:
            self.epochs += 1
            self.store_game_info()
            
        if self.epochs == max_epochs - 1:
            np.save(f"trained_q_matrices/trained_array_{max_epochs}", self.q_matrix)
            quit_game()
            
####################
#   Game functions 
####################
def draw_values():
    """ Draws epsilon, alpha, best score and number of epochs to game screen."""
    values_y = int(CELL_SIZE * CELL_NUMBER + 50)
    
    epsilon_text = f" = {robot.epsilon:.2f}"
    epsilon_surface = game_font.render(epsilon_text, True, (10, 10, 10))
    epsilon_x = 110
    epsilon_rect = epsilon_surface.get_rect(center=(epsilon_x, values_y))
    epsilon_img_rect = epsilon_img.get_rect(midright=(epsilon_rect.midleft))
    
    alpha_text = f" = {robot.alpha:.2f}"
    alpha_x = 280 
    alpha_img_rect = alpha_img.get_rect(midright=(alpha_x, values_y))
    alpha_surface = game_font.render(alpha_text, True, (10, 10, 10))
    alpha_rect = alpha_surface.get_rect(midleft=(alpha_img_rect.midright))

    best_score_text = f"Best: {robot.best_score}"
    best_score_x = 450
    best_score_surface = game_font.render(best_score_text, True, (10, 10, 10))
    best_score_rect = best_score_surface.get_rect(midleft=(best_score_x, values_y))

    epochs_text = f"Epochs: {robot.epochs}"
    epochs_x = int(CELL_SIZE * CELL_NUMBER - 200)
    epochs_surface = game_font.render(epochs_text, True, (10, 10, 10))
    epochs_rect = epochs_surface.get_rect(midleft=(epochs_x, values_y))
    
    screen.blit(epsilon_surface, epsilon_rect)
    screen.blit(epsilon_img, epsilon_img_rect )
    screen.blit(alpha_surface, alpha_rect)
    screen.blit(alpha_img, alpha_img_rect)
    screen.blit(best_score_surface, best_score_rect)
    screen.blit(epochs_surface, epochs_rect)

def quit_game():
    """ Quits the game, saves the best path, prints best score and plots the score over epochs"""
    np.save(f"paths/{robot.best_score}_points",robot.best_path)
    print(f"Best score: {robot.best_score}")
    pygame.quit()
    robot.plot_graph()
    sys.exit()




if __name__ == "__main__":


####################
#   Initialisation 
####################
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    clock = pygame.time.Clock()
    apple = pygame.image.load('graphics/apple.png').convert_alpha()
    epsilon_img = pygame.image.load('graphics/epsilon.png').convert_alpha()
    alpha_img = pygame.image.load('graphics/alpha.png').convert_alpha()
    game_font = pygame.font.Font('fonts/PlaypenSans-Regular.ttf', 25)

    SCREEN_UPDATE = pygame.USEREVENT
    pygame.time.set_timer(SCREEN_UPDATE, 1)
    time = 1
    pygame.display.set_caption('Snake - Train AI')
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
                robot.q_learning(2500)
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