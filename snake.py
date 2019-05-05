import pygame
import numpy as np
from random import sample


class Snake:
    def __init__(self, length, width, height, step):
        self.body = np.array([(x, 5) for x in range(5 + length, 5, -1)])
        self.width = width
        self.height = height
        self.step = step
        self.direction = 2
        self.directions = np.array([(0, -1), (0, 1), (1, 0), (-1, 0)])
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
        head = self.body[0]
        self_crash = (head == self.body[1:]).all(1).any()
        wall_crash = head[0] == 0 or head[0] == self.width or head[1] == 0 or head[1] == self.height
        return self_crash or wall_crash

    def move(self, direction):
        if (self.direction < 2 and direction > 1) or (self.direction > 1 and direction < 2):
            self.direction = direction

    def draw(self, surface, image):
        for i in range(len(self.body)):
            surface.blit(image, self.body[i] * self.step)


class Apple:
    def __init__(self, width, height, step):
        self.width = width
        self.height = height
        self.step = step
        self.pos = np.array((.8 * width, .8 * height))
        self.possible = {(i,j) for i in range(width) for j in range(height)}

    def new_pos(self, snake_body):
        self.pos = np.array(sample(self.possible-set(tuple(x) for x in snake_body), 1)[0])

    def draw(self, surface, image):
        surface.blit(image, self.pos * self.step)


class Game:
    def __init__(self, ai=True, width=25, height=25, title='Snake Game', length=4, step=16):
        self.ai = ai
        self.snake = Snake(length, width, height, step)
        self.apple = Apple(width, height, step)
        self.score = length * -10
        self.length = length
        self.width = width
        self.height = height
        self.step = step
        if not ai:
            pygame.init()
            pygame.display.set_caption(title)
            self.display_surf = pygame.display.set_mode((width * step, height * step), pygame.HWSURFACE)
            self.snake_image = pygame.image.load('resources/snake.png').convert()
            self.apple_image = pygame.image.load('resources/apple.png').convert()
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 20)
            str_start = pygame.font.Font(None, width).render('PRESS ENTER TO START', True, (255,255,255, 0.3))
            self.display_surf.blit(str_start, (width//3*step, height//2*step))
            pygame.display.update()
            start = False
            while not start:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RETURN:
                            start = True
                            break


    def restart(self):
        self.score = self.length * -10
        self.snake = Snake(self.length, self.width, self.height, self.step)
        self.apple = Apple(self.width, self.height, self.step)

    def render(self):
        self.display_surf.fill((0, 0, 0))
        self.snake.draw(self.display_surf, self.snake_image)
        self.apple.draw(self.display_surf, self.apple_image)
        # str_score = self.font.render('Score: {}'.format(self.getScore()), True, (255,255,255, 0.3))
        # self.display_surf.blit(str_score, (5, 5))
        pygame.display.flip()
        pygame.display.update()

    def get_score(self):
        return int(self.score + len(self.snake.body) * 10)

    def get_image(self):
        image = pygame.surfarray.array3d(self.display_surf)  # .swapaxes(0,1)
        # image = pygame.PixelArray(self.display_surf)
        return image.transpose((2, 1, 0))

    def execute(self, move=2):
        self.score += .05
        if np.array_equal(self.snake.body[0], self.apple.pos):
            self.snake.grow()
            self.apple.new_pos(self.snake.body)
        if self.snake.check_crash():
            return self.get_score()
        if self.ai:
            self.snake.move(move)
            self.snake.update()
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == 27:
                        return self.get_score()
                    self.snake.move(event.key - 273)
            self.snake.update()
            self.render()
            self.clock.tick(20)
        return False


if __name__ == '__main__':
    game = Game(ai=False)
    points = False
    while points is False:
        points = game.execute()
    print(points)
