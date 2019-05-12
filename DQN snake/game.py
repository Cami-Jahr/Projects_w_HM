import pygame
import numpy as np
from random import sample
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
        self.length = length
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
        self.apple = Apple(width, height, apple_image)

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
        self.score = self.length * -10
        self.snake.restart()
        self.apple.restart()
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
            self.clock.tick(20)

    def distance_head_to_apple(self):
        """Using manhattan distance, euclidean distance might be better"""
        return abs(self.snake.body[0][0] - self.apple.pos[0]) + abs(self.snake.body[0][1] - self.apple.pos[1])

    def get_score(self):
        if self.score:
            return self.score
        return self.distance_reward_value

    def get_image(self):
        """
        Get a image of the current game state as a (width, height, 3) ndarray
        representing the rbg colours of each pixel
        """
        # print(pygame.surfarray.array3d(self.game_logic_surf).transpose((2, 1, 0)))
        return pygame.surfarray.array3d(self.game_logic_surf).transpose((2, 1, 0))

    def execute(self, move=2):
        """
        Execute the input move if ai, else pygame event, and render outcome
        :param move: int in [0, 3]
        :return: score : int if game is over, else False
        """
        self.score = 0
        if self.ai:
            self.snake.move(move)
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == 27:
                        return False, self.get_score()
                    self.snake.move(event.key - 273)
        self.snake.update()
        if np.array_equal(self.snake.body[0], self.apple.pos):
            self.score = 1
            self.snake.grow()
            self.apple.new_pos(self.snake.body)

        new_distance = self.distance_head_to_apple()
        # Calculate distance reward as defined in Wei et al.
        self.distance_reward_value = log((self.length + self.prev_distance_head_to_apple) / (self.length + new_distance), self.length)
        self.prev_distance_head_to_apple = new_distance
        self.render()
        if self.snake.check_crash():
            self.score = -1
            return True, self.get_score()
        return False, self.get_score()


class Snake:
    """To handle the snake logic and render"""

    def __init__(self, length, width, height, head_image, body_image):
        self.body = np.array([(x, 2) for x in range(2 + length, 2, -1)])
        self.width = width
        self.height = height
        self.head_image = head_image
        self.body_image = body_image
        self.direction = 2
        self.directions = np.array([(0, -1), (0, 1), (1, 0), (-1, 0)])
        self.grow_pos = False
        self.length = length

    def restart(self):
        self.body = np.array([(x, 2) for x in range(2 + self.length, 2, -1)])
        self.direction = 2
        self.grow_pos = False

    def update(self):
        if self.grow_pos:
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

    def __init__(self, width, height, image):
        self.width = width
        self.height = height
        self.image = image
        self.pos = np.array((int(.8 * width), int(.8 * height)))
        self.possible = {(i, j) for i in range(width) for j in range(height)}

    def restart(self):
        self.pos = np.array((int(.8 * self.width), int(.8 * self.height)))
        self.possible = {(i, j) for i in range(self.width) for j in range(self.height)}

    def new_pos(self, snake_body):
        self.pos = np.array(sample(self.possible - set(tuple(x) for x in snake_body), 1)[0])

    def draw(self, surface):
        surface.blit(self.image, self.pos)


if __name__ == '__main__':
    game = Game(ai=False, width=14, height=14, render=True)
    points = done = False
    while done is False:
        done, points = game.execute()
    print(points)
