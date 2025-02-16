from pgzero.actor import Actor
import pygame


class Fruit:
    def __init__(self, name: str, is_bomb: bool, pos, vx, vy):
        self.name = name
        self.is_bomb = is_bomb
        self.actor = Actor("bomb" if is_bomb else name, pos)
        self.vx = vx
        self.vy = vy
        self.__sliced__ = False

    def updateActor(self, gravity):
        self.actor.x += self.vx
        self.actor.y += self.vy
        self.vy += gravity

    def getSliced(self):
        return self.__sliced__

    def setSliced(self):
        self.actor = Actor(self.name + "_sliced", (self.actor.x, self.actor.y))
        self.__sliced__ = True


class Splash:
    def __init__(self, pos):
        self.actor = Actor("splash_opaque", pos)
        self.duration = 90  # lasts 90 frames (1.5 sec at 60fps)

    def update(self):
        self.duration -= 1

class TrackedFruit(Fruit):
    def __init__(self, name, is_bomb, pos, vx, vy):
        super().__init__(name, is_bomb, pos, vx, vy)
        self.spawn_time = pygame.time.get_ticks()
