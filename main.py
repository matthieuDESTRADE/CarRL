import numpy as np
import pygame as pg
import noise
import cv2

# Create a Pygame window
pg.init()
screen = pg.display.set_mode((1000, 1000))
pg.display.set_caption("My Pygame Window")
font_style = pg.font.SysFont(None, 36)



carimg = pg.image.load('car.png').convert_alpha()
carimg = pg.transform.scale(carimg, (120, 120))

# Generate Perlin noise
width, height = 1000, 1050
scale = 0.001
octaves = 3
persistence = 0.5
lacunarity = 0.1
subs = 20

noise_map = [0] * (height // subs)
t = np.random.rand() * 1000

def fillnoise(t):
    for x in range(height // subs):
        nx = t + x * subs
        noise_value = noise.pnoise1(nx * scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity) * min(1000,nx)/1000 +0.3
        noise_map[x] = noise_value


# Car position and speed
car_x = 500
car_y = 650
car_speed = 20
angle = np.pi
speedval = 0
tv = 0.005

# Main game loop
running = True

up = False
down = False
l = False
r = False
pg.key.set_repeat(50, 50)

clock = pg.time.Clock()

lnoise = [(300 + 500 * noise_map[height // subs - 1 - x], x * subs) for x in range(height // subs)]
score = 0
time = 0
lastpoint = list(lnoise[np.argmin((car_x-np.array(lnoise)[:,0])**2+(car_y-np.array(lnoise)[:,1])**2)])

Vjeu = 1

while running:
    clock.tick(30)
    dt = clock.get_time() * Vjeu
    time += dt

    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    keys = pg.key.get_pressed()
    up = keys[pg.K_UP]
    down = keys[pg.K_DOWN]
    l = keys[pg.K_LEFT]
    r = keys[pg.K_RIGHT]

    if up:
        speedval += .0005 * dt
    if down:
        speedval -= .0005 * dt
    if l:
        angle += tv * dt * min(speedval,2)/2
    if r:
        angle -= tv*dt* min(speedval,2)/2

    car_x = car_x + speedval * np.sin(angle) * dt
    car_y = car_y + speedval * np.cos(angle) * dt

    lnoise = [(300 + 500 * noise_map[height // subs - 1 - x], x * subs) for x in range(height // subs)]
    dist = np.min((car_x-np.array(lnoise)[:,0])**2+(car_y-np.array(lnoise)[:,1])**2)
    if np.sqrt(dist) < 110:
        speedval *= 0.9992**dt
    else:
        speedval *= 0.992**dt


    intpt = np.argmin((car_x-np.array(lnoise)[:,0])**2+(car_y-np.array(lnoise)[:,1])**2)
    curpoint = list(lnoise[intpt])
    curpoint[1] -= t
    if list(lnoise[(intpt+1)%len(lnoise)])[1] <= car_y +t and curpoint[1] >= car_y:
        lastpoint = list(lnoise[(intpt+1)%len(lnoise)])
        lastpoint[1] -= t
        vec = np.array(curpoint) - np.array(lastpoint)

    else:
        lastpoint = list(lnoise[(intpt-1)])
        lastpoint[1] -= t
        vec = np.array(lastpoint) - np.array(curpoint)  



    norm = np.sqrt((lastpoint[0]-curpoint[0])**2 + (lastpoint[1]-curpoint[1])**2)**0.5
    rwrd = 0.1/5*(vec[0]*(speedval * np.sin(angle) * dt) + vec[1]*(speedval * np.cos(angle) * dt))/norm
    rwrd += -0.2*(0.1/5)*(vec[0]*(speedval * np.cos(angle) * dt) - vec[1]*(speedval * np.sin(angle) * dt))*np.sign(vec[0]*(car_y-t-curpoint[1]) - vec[1]*(car_x-curpoint[0]))/norm

    score += rwrd


    # Update game logic here
    # Draw to the screen here
    screen.fill((10, 150, 50))  # Clear the screen with white color

    # pg.draw.lines(screen, (100, 100, 100), False, [(300 + 500 * noise_map[height//subs-1-x],x*subs) for x in range(height//subs)], 200)
    for x in range(height // subs):
        pg.draw.circle(screen, (100, 100, 100), (300 + 500 * noise_map[height // subs - 1 - x], x * subs), 100)

    pg.draw.lines(screen, (255, 255, 255), False,lnoise, 5)

    # Draw the car
    rotated_car = pg.transform.rotate(carimg, (angle+np.pi)/np.pi*180)
    screen.blit(rotated_car, rotated_car.get_rect(center=carimg.get_rect(topleft=(car_x-carimg.get_size()[0]/2, car_y-carimg.get_size()[1]/2)).center).topleft)


    # Create a text surface
    text_surface = font_style.render("score : " + str(rwrd)+" m", True, (255, 255, 255))
    text_surface2 = font_style.render("time : " + str(time/1000)+" s", True, (255, 255, 255))
    ms = (score/5) / (time/1000) if time>0 else 0
    text_surface3 = font_style.render("mean speed : " + str(int(100*ms)/100)+" m/s", True, (255, 255, 255))


    screen.blit(text_surface, (10,10))
    screen.blit(text_surface2, (10,50))
    screen.blit(text_surface3, (10,90))

    # Move the road
    lim = 600
    lim2 = 800
    v = 0.005
    if car_y < lim:
        fac = (lim - car_y) * v * dt
        t += fac
        car_y += fac

    if car_y > lim2:
        fac = (car_y - lim2) * v * dt
        t -= fac
        car_y -= fac

    fillnoise(t)

    if time/1000 > 60:
        running = False 

    pg.display.flip()

    image = pg.surfarray.array3d(screen) 

    resized_image = cv2.resize(image, (10, 10))
    #cv2.imshow("Resized Image", resized_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    

print("Score : ", score, "m")
print("Mean speed : ", int(100*ms)/100, "m/s")
pg.quit()
