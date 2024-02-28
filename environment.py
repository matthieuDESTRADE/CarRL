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
        noise_map = [0] * (self.height // self.subs)

        for x in range(self.height // self.subs):
            nx = t + x * self.subs
            noise_value = 0.8*noise.pnoise1(nx * scale, octaves=octaves, persistence=persistence,
                                            lacunarity=lacunarity) * min(5000, nx-self.tinit)/5000 + 0.3
            noise_map[x] = noise_value
        return noise_map

    def __init__(self, maxtime=60, display_screen=True, evaluation=False):
        super().__init__()

        self.display_screen = display_screen
        self.maxtime = maxtime

        self.time = 0
        self.score = 0

        pg.init()

        self.screen = pg.Surface((1000, 1000))
        self.font_style = pg.font.SysFont(None, 36)

        if display_screen:
            self.dsplay = pg.display.set_mode((1000, 1000))
            pg.display.set_caption("My Pygame Window")
        else:
            pg.display.set_mode((1, 1), 0, 32)

        self.carimg = pg.image.load('car.png').convert_alpha()
        self.carimg = pg.transform.scale(self.carimg, (120, 120))

        self.width, self.height = 1000, 1050
        self.subs = 20

        # Car position and speed
        self.car_x = 500
        self.car_y = 650
        self.car_speed = 20
        self.evaluation = evaluation
        if evaluation:
            self.angle = np.pi
        else:
            self.angle = np.pi + np.random.rand()*2*np.pi*0.1

        self.speedval = 0
        self.tv = 0.005

        self.t = np.random.rand()*10000000
        self.tinit = self.t
        self.noise_map = self.fillnoise(self.t)
        lnoise = [(300 + 500 * self.noise_map[self.height // self.subs - 1 - x],
                   x * self.subs) for x in range(self.height // self.subs)]

        self.lastpoint = list(lnoise[np.argmin(
            (self.car_x-np.array(lnoise)[:, 0])**2+(self.car_y-np.array(lnoise)[:, 1])**2)])

        # Define the action and observation spaces
        self.action_space = spaces.Discrete(4) # TODO: change to 6
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)

        if not (display_screen):
            pg.quit()

    def reset(self):
        # Reset the environment and return the initial observation (frame)
        self.__init__(self.maxtime, self.display_screen,
                      self.evaluation)  # Reinitialize the environment
        observation = self._get_observation()
        return observation

    def step(self, action, dt=1000/30):
        # Perform the given action and return the new observation, reward, and done flag

        done = False
        self.time += dt
        rwrd = 0

        if action == 0 or action == 2 or action == 3:
            self.speedval += .0005 * dt
        if action == 1 or action == 4 or action == 5:
            self.speedval -= .0005 * dt
        if action == 2 or action == 4:
            self.angle += self.tv * dt * min(self.speedval, 2)/2
        if action == 3 or action == 5:
            self.angle -= self.tv*dt * min(self.speedval, 2)/2

        self.car_x = self.car_x + self.speedval * np.sin(self.angle) * dt
        self.car_y = self.car_y + self.speedval * np.cos(self.angle) * dt

        self.noise_map = self.fillnoise(self.t)
        lnoise = [(300 + 500 * self.noise_map[self.height // self.subs - 1 - x],
                   x * self.subs) for x in range(self.height // self.subs)]
        dist = np.min((self.car_x-np.array(lnoise)
                      [:, 0])**2+(self.car_y-np.array(lnoise)[:, 1])**2)
        if np.sqrt(dist) < 110:
            self.speedval *= 0.9992**dt
        else:
            self.speedval *= 0.992**dt
            # done = True

        intpt = np.argmin((self.car_x-np.array(lnoise)
                          [:, 0])**2+(self.car_y-np.array(lnoise)[:, 1])**2)
        curpoint = list(lnoise[intpt])
        curpoint[1] -= self.t
        if list(lnoise[(intpt+1) % len(lnoise)])[1] <= self.car_y + self.t and curpoint[1] >= self.car_y:
            lastpoint = list(lnoise[(intpt+1) % len(lnoise)])
            lastpoint[1] -= self.t
            vec = np.array(curpoint) - np.array(lastpoint)

        else:
            lastpoint = list(lnoise[(intpt-1)])
            lastpoint[1] -= self.t
            vec = np.array(lastpoint) - np.array(curpoint)

        norm = np.sqrt((lastpoint[0]-curpoint[0]) **
                       2 + (lastpoint[1]-curpoint[1])**2)**0.5
        # @matt : isnt this the same as np.linalg.norm(vec) ?
        rwrd = 0.1/5*(vec[0]*(self.speedval * np.sin(self.angle) * dt) +
                      vec[1]*(self.speedval * np.cos(self.angle) * dt))/norm
        self.score += rwrd
        # rwrd += -(0.1/5)*(vec[0]*(self.speedval * np.cos(self.angle) * dt) - vec[1]*(self.speedval * np.sin(self.angle) * dt))*np.sign(vec[0]*(self.car_y-self.t-curpoint[1]) - vec[1]*(self.car_x-curpoint[0]))/norm

        if np.sqrt(dist) > 120:
            done = True

        # Update game logic here
        # Draw to the screen here

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

        return obs, rwrd, done, {}

    def _get_observation(self):
        # Update game logic here
        # Draw to the screen here
        self.screen.fill((10, 150, 50))  # Clear the screen with white color

        # pg.draw.lines(screen, (100, 100, 100), False, [(300 + 500 * noise_map[height//subs-1-x],x*subs) for x in range(height//subs)], 200)
        for x in range(self.height // self.subs):
            pg.draw.circle(self.screen, (100, 100, 100), (300 + 500 *
                           self.noise_map[self.height // self.subs - 1 - x], x * self.subs), 100)

        lnoise = [(300 + 500 * self.noise_map[self.height // self.subs - 1 - x],
                   x * self.subs) for x in range(self.height // self.subs)]
        # pg.draw.lines(self.screen, (255, 255, 0), False,lnoise, 5)

        # Draw the car
        rotated_car = pg.transform.rotate(
            self.carimg, (self.angle+np.pi)/np.pi*180)
        self.screen.blit(rotated_car, rotated_car.get_rect(center=self.carimg.get_rect(topleft=(
            self.car_x-self.carimg.get_size()[0]/2, self.car_y-self.carimg.get_size()[1]/2)).center).topleft)

        if self.display_screen:
            # Create a text surface

            self.dsplay.blit(self.screen, (0, 0))

            text_surface = self.font_style.render(
                "score : " + str(int(self.score))+" m", True, (255, 255, 255))
            text_surface2 = self.font_style.render(
                "time : " + str(self.time/1000)+" s", True, (255, 255, 255))
            self.ms = (self.score/5) / (self.time/1000) if self.time > 0 else 0
            text_surface3 = self.font_style.render(
                "mean speed : " + str(int(100*self.ms)/100)+" m/s", True, (255, 255, 255))

            self.dsplay.blit(text_surface, (10, 10))
            self.dsplay.blit(text_surface2, (10, 50))
            self.dsplay.blit(text_surface3, (10, 90))

            pg.display.flip()

        # Capture the current frame

        # rotated_screen_data = pg.transform.rotate(self.screen, np.random.rand()*360)

        image = pg.surfarray.array3d(self.screen)
        resized_image = cv2.resize(
            image, (40, 40), interpolation=cv2.INTER_AREA)
        # cropped_image = resized_image[15:65, 15:65]

        # Rotate the screen data (example rotation by 45 degrees)
        # rotated_screen_data = pg.transform.rotate(self.screen, -self.angle/np.pi*180)
        # Rotate the car coordinates
        # rotated_car_x = self.car_x * np.cos(self.angle) - self.car_y * np.sin(self.angle)
        # rotated_car_y = self.car_x * np.sin(self.angle) + self.car_y * np.cos(self.angle)

        return resized_image

    @staticmethod
    def obs2tensor(obs, device=None):
        if not device:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not (isinstance(obs[0, 0, 0], np.uint8)):
            return torch.tensor(obs, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        else:
            return torch.tensor(obs, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(device)
