import pygame
import numpy as np
from random import sample, randint
from math import log


class Game:
    """Main snake game class, to handle input commands, game rendering, score calculation and image creation"""

    def __init__(self, ai=True, width=25, height=25, title='Snake Game', length=3, fit_to_size=600, render=False):
        """
        :param ai: If ai or human input to be used
        :param width: width of board, minimum 7
        :param height: height of board, minimum 5, same as width if ai is running
        :param title: name of game window
        :param length: initial length of snake
        :param fit_to_size: width of the game window it render=True.
        Dictates the width, with fit_to_size being the maximum size in pixels
        :param render: if True the gamewindow is displayed, if False Game.get_image() will need to be called to get
        progress information
        """
        if width < 7:
            width = 7
        if height < 5:
            height = 5
        if ai:
            height = width  # Assure cubic from game itself if ai
        self.ai = ai
        self.score = 0
        self.score_balancer = -length
        self.ate = False
        self.width = width
        self.height = height
        self.pixels_per_unit = fit_to_size // width
        self.should_render = render
        self.distance_reward_value = 0
        if render:
            pygame.init()
            pygame.display.set_caption(title)
        self.display_surf = pygame.display.set_mode((width * self.pixels_per_unit, height * self.pixels_per_unit))
        self.game_logic_surf = pygame.Surface((width, height))

        head_image = pygame.image.load('resources/head.png').convert()
        body_image = pygame.image.load('resources/snake.png').convert()
        apple_image = pygame.image.load('resources/apple.png').convert()
        self.snake = Snake(length, width, height, head_image, body_image)
        self.apple = Apple(width, height, apple_image, self.snake.body)

        self.prev_distance_head_to_apple = self.distance_head_to_apple()

        self.clock = pygame.time.Clock()
        self.render()
        if not ai and render:
            str_start = pygame.font.Font(None, (width * self.pixels_per_unit) // 20) \
                .render('PRESS ENTER TO START', True, (255, 255, 255, 0.3))
            self.display_surf.blit(str_start,
                                   ((width * self.pixels_per_unit) // 3, (self.pixels_per_unit * height) // 2))
            pygame.display.update()
            start = False
            while not start:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RETURN:
                            start = True
                            break

    def __call__(self, *args, **kwargs):
        return self.execute(*args)

    @staticmethod
    def get_amount_of_legal_actions():
        return 4

    def restart(self):
        self.snake.restart()
        self.apple.restart(self.snake.body)
        self.render()

    def render(self):
        self.game_logic_surf.fill((0, 0, 0))
        self.snake.draw(self.game_logic_surf)
        self.apple.draw(self.game_logic_surf)
        if self.should_render:
            pygame.transform.scale(self.game_logic_surf,
                                   (self.width * self.pixels_per_unit, self.height * self.pixels_per_unit),
                                   self.display_surf)
            pygame.display.flip()
            pygame.display.update()
            self.clock.tick(30)

    def distance_head_to_apple(self):
        """Using manhattan distance, euclidean distance might be better"""
        return abs(self.snake.body[0][0] - self.apple.pos[0]) + abs(self.snake.body[0][1] - self.apple.pos[1])

    def distance_value(self, new_distance):
        return log(
            (self.snake.current_length + self.prev_distance_head_to_apple) / (self.snake.current_length + new_distance),
            self.snake.current_length)

    def get_turn_score(self):
        return self.score + self.distance_reward_value

    def get_final_score(self):
        return self.score_balancer + self.snake.current_length

    def get_image(self):
        """
        Get a image of the current game state as a (width, height, 3) ndarray
        representing the rbg colours of each pixel
        """
        return pygame.surfarray.array3d(self.game_logic_surf).transpose((2, 1, 0))

    def execute(self, move=2):
        """
        Execute the input move if ai, else pygame event, and render outcome
        :param move: int in [0, 3]
        :return: (done, length, score)
        """
        self.score = 0
        self.ate = False
        if self.ai:
            self.snake.move(move)
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == 27:
                        return self.turn_result(False)
                    self.snake.move(event.key - 273)
        self.snake.update()
        if np.array_equal(self.snake.body[0], self.apple.pos):
            self.score = 1
            self.ate = True
            self.snake.grow()
            self.apple.new_pos(self.snake.body)

        new_distance = self.distance_head_to_apple()
        # Calculate distance reward as defined in Wei et al.
        self.distance_reward_value = self.distance_value(new_distance)
        self.prev_distance_head_to_apple = new_distance
        self.render()
        if self.snake.check_crash():
            self.score = -1
            return self.turn_result(True)
        return self.turn_result(False)

    def turn_result(self, crashed):
        return crashed, self.ate, self.snake.current_length, self.get_turn_score()


class Snake:
    """To handle the snake logic and render"""

    def __init__(self, length, width, height, head_image, body_image):
        x_head = randint(1 + length, width - 4)
        y_cord = randint(2, height - 2)
        self.body = np.array([(x, y_cord) for x in range(x_head, x_head - length, -1)])
        self.width = width
        self.height = height
        self.head_image = head_image
        self.body_image = body_image
        self.direction = 2
        self.directions = np.array([(0, -1), (0, 1), (1, 0), (-1, 0)])
        self.grow_pos = False
        self.length = length
        self.current_length = length

    def restart(self):
        x_head = randint(1 + self.length, self.width - 4)
        y_cord = randint(2, self.height - 2)
        self.body = np.array([(x, y_cord) for x in range(x_head, x_head - self.length, -1)])
        self.direction = 2
        self.grow_pos = False
        self.current_length = self.length

    def update(self):
        if self.grow_pos:
            self.current_length += 1
            self.body = np.vstack([self.body[0] + self.directions[self.direction], self.body])
            self.grow_pos = False
        else:
            self.body = np.vstack([self.body[0] + self.directions[self.direction], self.body[:-1]])

    def grow(self):
        self.grow_pos = True

    def check_crash(self):
        """Should be called after movement has been calculated"""
        head = self.body[0]
        self_crash = (head == self.body[1:]).all(1).any()
        wall_crash = head[0] == -1 or head[0] == self.width or head[1] == -1 or head[1] == self.height
        return self_crash or wall_crash

    def move(self, direction: int) -> None:
        """
        :param direction: direction the snake should move if possible (cannot turn 180 degrees)
        0 = Up
        1 = Down
        2 = Right
        3 = Left
        """
        if (self.direction < 2 and direction > 1) or (self.direction > 1 and direction < 2):
            self.direction = direction

    def draw(self, surface):
        surface.blit(self.head_image, self.body[0])
        for i in range(1, len(self.body)):
            surface.blit(self.body_image, self.body[i])


class Apple:
    """To handle the apple position and render"""

    def __init__(self, width, height, image, snake_body):
        self.width = width
        self.height = height
        self.image = image
        self.possible = {(i, j) for i in range(width) for j in range(height)}
        self.pos = np.array(sample(self.possible - set(tuple(x) for x in snake_body), 1)[0])

    def restart(self, snake_body):
        self.pos = np.array(sample(self.possible - set(tuple(x) for x in snake_body), 1)[0])

    def new_pos(self, snake_body):
        self.pos = np.array(sample(self.possible - set(tuple(x) for x in snake_body), 1)[0])

    def draw(self, surface):
        surface.blit(self.image, self.pos)


if __name__ == '__main__':
    game = Game(ai=False, width=14, height=14, render=True)
    while not game.execute()[0]:
        pass
    print(game.get_final_score())
