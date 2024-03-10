import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame as pg
import noise
import cv2
import torch


class CarEnv(gym.Env):
    def fillnoise(self, t):
        # Generate Perlin noise
        scale = 0.001
        octaves = 3
        persistence = 0.5
        lacunarity = 0.1
        noise_map = [0] * (self.display_height // self.subs)

        for x in range(self.display_height // self.subs):
            nx = t + x * self.subs
            noise_value = 0.8*noise.pnoise1(nx * scale, octaves=octaves, persistence=persistence,
                                            lacunarity=lacunarity) * min(5000, nx-self.tinit)/5000 + 0.3
            noise_map[x] = noise_value

        self.lnoise = np.array([(300 + 500 * noise_map[self.display_height // self.subs - 1 - x],
                                 x * self.subs) for x in range(self.display_height // self.subs)])

    def __init__(self, maxtime=60, display=True, evaluation=False, draw_central_line=False):
        super().__init__()

        self.draw_central_line = draw_central_line
        self.evaluation = evaluation
        self.display = display
        self.maxtime = maxtime

        self.time = 0
        self.score = 0

        self.display_width, self.display_height = 1000, 1000
        size = (self.display_width, self.display_height)

        self.observation_size = (40, 40, 3)

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=self.observation_size, dtype=np.uint8)

        self.subs = 20

        # Car position and speed
        self.car_x = 500
        self.car_y = 650
        self.car_speed = .0005
        self.angle = np.pi
        self.maxspeed = True

        # random angle for training
        if not self.evaluation:
            self.angle += np.random.rand()*2*np.pi*0.1

        self.speedval = 0
        self.tv = 0.005

        # noise
        self.t = np.random.rand()*10000000
        self.tinit = self.t
        self.fillnoise(self.t)

        if not self.display:
            size = (1, 1)

        pg.init()
        self.display_screen = pg.display.set_mode(size)
        self.screen = pg.Surface(size)
        carimg = pg.image.load('car.png').convert_alpha()
        self.carimg = pg.transform.scale(carimg, (120, 120))
        self.font_style = pg.font.SysFont(None, 36)
        pg.display.set_caption("My Pygame Window")

    def reset(self):
        # Reset the environment and return the initial observation (frame)
        self.__init__(self.maxtime,
                      self.display,
                      self.evaluation,
                      self.draw_central_line
                      )
        observation = self._get_observation()
        return observation

    def step(self, action, dt=1000/30):
        # Perform the given action and return the new observation, reward, and done flag

        done = False
        self.time += dt
        rwrd = 0

        if action == 0 or action == 2 or action == 3:
            self.speedval += self.car_speed * dt
        if action == 1 or action == 4 or action == 5:
            self.speedval -= self.car_speed * dt
        if action == 2 or action == 4:
            self.angle += self.tv * dt * min(self.speedval, 2)/2
        if action == 3 or action == 5:
            self.angle -= self.tv*dt * min(self.speedval, 2)/2

        self.car_x = self.car_x + self.speedval * np.sin(self.angle) * dt
        self.car_y = self.car_y + self.speedval * np.cos(self.angle) * dt

        pos = np.array([self.car_x, self.car_y])
        noise_dist = np.linalg.norm(pos - self.lnoise, axis=1)
        dist = np.min(noise_dist)
        if dist < 110:
            if self.maxspeed:
                self.speedval *= 0.9992**dt
        else:
            self.speedval *= 0.992**dt


        intpt = np.argmin(noise_dist)
        curpoint = self.lnoise[intpt]
        curpoint[1] -= self.t
        if self.lnoise[(intpt+1) % len(self.lnoise)][1] <= self.car_y + self.t and curpoint[1] >= self.car_y:
            lastpoint = self.lnoise[(intpt+1) % len(self.lnoise)]
            lastpoint[1] -= self.t
            vec = np.array(curpoint) - np.array(lastpoint)

        else:
            lastpoint = self.lnoise[(intpt-1)]
            lastpoint[1] -= self.t
            vec = lastpoint - curpoint

        norm = np.linalg.norm(vec)

        rwrd = 0.1/5*(vec[0]*(self.speedval * np.sin(self.angle) * dt) +
                      vec[1]*(self.speedval * np.cos(self.angle) * dt))/norm
        self.score += rwrd
        # rwrd += -(0.1/5)*(vec[0]*(self.speedval * np.cos(self.angle) * dt) - vec[1]*(self.speedval * np.sin(self.angle) * dt))*np.sign(vec[0]*(self.car_y-self.t-curpoint[1]) - vec[1]*(self.car_x-curpoint[0]))/norm

        if dist > 120:
            done = True

        # Move the road
        lim = 600
        lim2 = 800
        v = 0.005
        if self.car_y < lim:
            fac = (lim - self.car_y) * v * dt
            self.t += fac
            self.car_y += fac

        if self.car_y > lim2:
            fac = (self.car_y - lim2) * v * dt
            self.t -= fac
            self.car_y -= fac

        self.fillnoise(self.t)

        obs = self._get_observation()
        if self.time/1000 > self.maxtime:
            done = True

        if done:
            pg.quit()

        if obs is None:
            # user closed the window
            done = True

        return obs, rwrd, done, {}

    def _get_observation(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                return None

        # Clear the screen with white color
        self.screen.fill((10, 150, 50))

        for x in range(self.display_height // self.subs):
            pg.draw.circle(self.screen, (100, 100, 100), self.lnoise[x], 100)

        if self.draw_central_line:
            pg.draw.lines(self.screen, (200, 200, 200), False, self.lnoise, 5)

        # Draw the car
        rotated_car = pg.transform.rotate(
            self.carimg, (self.angle+np.pi)/np.pi*180)
        self.screen.blit(rotated_car, rotated_car.get_rect(center=self.carimg.get_rect(topleft=(
            self.car_x-self.carimg.get_size()[0]/2, self.car_y-self.carimg.get_size()[1]/2)).center).topleft)

        if self.display:
            self.display_screen.blit(self.screen, (0, 0))

            mean_speed = (self.score/5) / (self.time /
                                           1000) if self.time > 0 else 0

            score_surface = self.font_style.render(
                "score : " + str(int(self.score))+" m", True, (255, 255, 255))
            time_surface = self.font_style.render(
                "time : " + str(self.time/1000)+" s", True, (255, 255, 255))
            speed_surface = self.font_style.render(
                "mean speed : " + str(round(mean_speed, 2))+" m/s", True, (255, 255, 255))

            self.display_screen.blit(score_surface, (10, 10))
            self.display_screen.blit(time_surface, (10, 50))
            self.display_screen.blit(speed_surface, (10, 90))

            pg.display.flip()

        # Capture the current frame
        image = pg.surfarray.array3d(self.screen)
        resized_image = cv2.resize(
            image, self.observation_size[:2], interpolation=cv2.INTER_AREA)
        # cropped_image = resized_image[15:65, 15:65]

        return resized_image

    @staticmethod
    def obs2tensor(obs, device=None):
        if not device:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        if not (isinstance(obs[0, 0, 0], np.uint8)):
            return torch.tensor(obs, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        else:
            return torch.tensor(obs, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(device)
