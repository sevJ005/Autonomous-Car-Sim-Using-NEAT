import pygame
import math
import neat
import os
import sys

# Window settings
WIDTH, HEIGHT = 1000, 600
CAR_SIZE = 30

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("NEAT Car Simulation")
clock = pygame.time.Clock()

# Load images
car_img = pygame.image.load("car.png").convert_alpha()
car_img = pygame.transform.scale(car_img, (CAR_SIZE, CAR_SIZE))

track_img = pygame.image.load("Spa.png").convert()
track = pygame.transform.scale(track_img, (WIDTH, HEIGHT))

# Border color
BORDER_COLOR = (127,127,127)

# Car class
class Car:
    def __init__(self):
        self.x = 100
        self.y = 300
        self.angle = -40
        self.speed = 5
        self.size = CAR_SIZE
        self.radars = []
        self.alive = True
        self.distance = 0

    def update(self, track, border_color):
        if not self.alive:
            return

        # Move forward
        self.x += math.cos(math.radians(self.angle)) * self.speed
        self.y += math.sin(math.radians(self.angle)) * self.speed
        self.distance += self.speed

        # Check border collision
        cx = int(self.x + self.size / 2)
        cy = int(self.y + self.size / 2)
        if 0 <= cx < WIDTH and 0 <= cy < HEIGHT:
            if track.get_at((cx, cy))[:3] == border_color:
                self.alive = False

        # Keep car inside window
        self.x = max(0, min(self.x, WIDTH - self.size))
        self.y = max(0, min(self.y, HEIGHT - self.size))

        # Update radars
        self.radars.clear()
        for d in [-90, -45, 0, 45, 90]:
            self.check_radar(d, track, border_color)

    def check_radar(self, degree, track, border_color):
        length = 0
        while length < 300:
            x = int(self.x + self.size/2 + math.cos(math.radians(self.angle + degree)) * length)
            y = int(self.y + self.size/2 + math.sin(math.radians(self.angle + degree)) * length)
            if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
                break
            if track.get_at((x, y))[:3] == border_color:
                break
            length += 1
        self.radars.append(length / 100)  # scale down for NEAT inputs

    def draw(self, screen):
        if not self.alive:
            return
        rotated = pygame.transform.rotate(car_img, -self.angle)
        rect = rotated.get_rect(center=(self.x + self.size/2, self.y + self.size/2))
        screen.blit(rotated, rect.topleft)
        # Draw radars
        for i, d in enumerate([-90, -45, 0, 45, 90]):
            length = self.radars[i] * 100
            end_x = int(self.x + self.size/2 + math.cos(math.radians(self.angle + d)) * length)
            end_y = int(self.y + self.size/2 + math.sin(math.radians(self.angle + d)) * length)
            pygame.draw.line(screen, (0,255,0), (self.x+self.size/2, self.y+self.size/2), (end_x, end_y), 1)
            pygame.draw.circle(screen, (0,255,0), (end_x, end_y), 3)

# NEAT evaluation function
def eval_genomes(genomes, config):
    cars = []
    nets = []
    ge = []

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        genome.fitness = 0
        ge.append(genome)
        cars.append(Car())

    run = True
    frame_count = 0
    while run and len(cars) > 0:
        clock.tick(60)
        frame_count += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.blit(track, (0, 0))

        alive_count = 0
        for i, car in enumerate(cars):
            if not car.alive:
                continue
            alive_count += 1

            prev_distance = car.distance
            car.update(track, BORDER_COLOR)

            # Penalize cars that barely move (spinning)
            if abs(car.distance - prev_distance) < 0.1:
                ge[i].fitness -= 0.1

            # Feed radar distances to neural network
            output = nets[i].activate(car.radars)
            steering = (output[0] - output[1]) * 10   # turning
            throttle = max(0, output[2]) * 5          # forward only
            car.angle += steering
            car.speed = max(0, min(throttle, 5))

            # Fitness: reward distance + bonus for fast finish
            ge[i].fitness += car.speed * 0.1  # reward movement
            ge[i].fitness += car.distance * 0.01  # reward covering track

            # Optional: if you have a finish line:
            # if car.finished_track:
            #     ge[i].fitness += 1000 / frame_count  # faster = higher reward

            car.draw(screen)

        # Display alive count
        font = pygame.font.SysFont("Arial", 20)
        text = font.render(f"Alive: {alive_count}", True, (0, 0, 0))
        screen.blit(text, (10, 10))

        # Stop evaluation if all cars crashed
        if alive_count == 0 or frame_count > 900:  # limit max frames for time 900/60 
            break

        pygame.display.update()


# Load NEAT config
config_path = "config.txt"
config = neat.config.Config(neat.DefaultGenome,
                            neat.DefaultReproduction,
                            neat.DefaultSpeciesSet,
                            neat.DefaultStagnation,
                            config_path)


# Create population and add reporters
p = neat.Population(config)
p.add_reporter(neat.StdOutReporter(True))
p.add_reporter(neat.StatisticsReporter())

# Run NEAT
p.run(eval_genomes, 50)  # 50 generations max

pygame.quit()
