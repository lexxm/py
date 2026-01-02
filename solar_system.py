import pygame  # Library for visualization
import math    # Mathematical functions
import csv  # For saving the satellite trajectory
import math
import copy
import numpy as np

#print(np.finfo(np.longdouble))

###########################################################
# Constants
###########################################################
WIDTH, HEIGHT = 2000, 1000  # Window size
#G = np.longdouble(6.6743015e-11)            # Gravitational constant
G = np.longdouble(6.6745e-11)
#SCALE = 3e-9#6e-10               # Scale for visualization (reduces distances)
SCALE = 2e-6
CENTER_X, CENTER_Y = 0, 0
TIMESTEP = 100#3600            # Time step (2 hour per frame)

# Colors for planets
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
BLUE = (100, 149, 237)
GRAY = (128, 128, 128)
RED = (188, 39, 50)
ORANGE = (255, 165, 0)
BROWN = (210, 180, 140)


pygame.init()  # Initialize pygame
win = pygame.display.set_mode((WIDTH, HEIGHT))  # Create window
pygame.display.set_caption("Solar System")

 # Class describing a celestial body (planet or star)
class Body:
    def check_collision(self, other):
        """Check for collision with another body using physical radii (in meters)."""
        if self.real_radius is None or other.real_radius is None:
            return False
        dx = other.x - self.x
        dy = other.y - self.y
        dist = math.sqrt(dx**2 + dy**2)
        return dist <= (self.real_radius + other.real_radius)
    
    def check_collision(self, endX, endY, other):
        """Check for collision with another body using physical radii (in meters)."""
        if self.real_radius is None or other.real_radius is None:
            return False
        
        dist = distance_point_to_line(other.x, other.y, self.x, self.y, endX, endY)
        if dist <= (self.real_radius + other.real_radius):
            return True
        return False

    def __init__(self, x, y, draw_radius, color, mass, vx=0, vy=0, has_rings=False, real_radius=None, visible=True, name=None):
        self.x = np.longdouble(x)              # X coordinate (m)
        self.y = np.longdouble(y)              # Y coordinate (m)
        self.draw_radius = draw_radius         # Radius for drawing (pixels)
        self.color = color                     # Color
        self.mass = np.longdouble(mass)        # Mass (kg)
        self.vx = np.longdouble(vx)            # Velocity in X (m/s)
        self.vy = np.longdouble(vy)            # Velocity in Y (m/s)
        self.orbit = []         # List of orbit points for visualization
        self.has_rings = has_rings  # Has rings
        self.real_radius = real_radius  # Real radius (m)
        self.visible = visible  # Body visibility
        self.name = name        # Name of the body
        # If orbital_radius wasn't provided, try to compute it from a global `sun` object (if present).
        #s = globals().get('sun', None)
        #if s is not None and hasattr(s, 'x') and hasattr(s, 'y'):
        self.orbital_radius = math.sqrt(x*x + y*y)  # Distance to central body (m), if set

    def draw(self, win):
        """Draw the body and its orbit on the screen"""
        if not self.visible:
            return
        x = (self.x-CENTER_X) * SCALE + WIDTH // 2
        y = (self.y-CENTER_Y) * SCALE + HEIGHT // 2
        # Draw rings for Saturn
        if self.has_rings:
            ring_color = (200, 200, 180)
            # Outer ellipse (ring)
            pygame.draw.ellipse(win, ring_color, [int(x - self.draw_radius*2.5), int(y - self.draw_radius*1.2), int(self.draw_radius*5), int(self.draw_radius*2.4)], 2)
            # Inner ellipse (thin part of the ring)
            pygame.draw.ellipse(win, ring_color, [int(x - self.draw_radius*1.5), int(y - self.draw_radius*0.7), int(self.draw_radius*3), int(self.draw_radius*1.4)], 1)
        #pygame.draw.circle(win, self.color, (int(x), int(y)), self.draw_radius)
        pygame.draw.circle(win, self.color, (int(x), int(y)), 1)
        pygame.draw.circle(win, self.color, (int(x), int(y)), self.real_radius*SCALE)
        # Draw name
        if self.name:
            font = pygame.font.SysFont('arial', 16)
            text = font.render(self.name, True, (255,255,255))
            win.blit(text, (int(x) + self.draw_radius + 5, int(y) - self.draw_radius - 5))
        # Draw orbit line with varying thickness
        n = len(self.orbit)
        #print(self.name, " orbit points: ", n)
        if n > 2:
            points = [((ox-CENTER_X) * SCALE + WIDTH // 2, (oy-CENTER_Y) * SCALE + HEIGHT // 2) for ox, oy in self.orbit]
            one_third = n // 3
            two_third = 2 * n // 3
            # First third (oldest points)
            
            if one_third > 1:
                pygame.draw.lines(win, self.color, False, points[:one_third], 1)
            # Middle third
            if two_third - one_third > 1:
                pygame.draw.lines(win, self.color, False, points[one_third:two_third], 2)
            # Last third (most recent points)
            if n - two_third > 1:
                pygame.draw.lines(win, self.color, False, points[two_third:], 3)

    def attract(self, other):
        """Calculates gravitational attraction between bodies and checks for collision"""
        dx = np.longdouble(other.x - self.x)
        dy = np.longdouble(other.y - self.y)
        dist = dx**2 + dy**2  # Distance between bodies
        if dist == 0:
            return 0, 0  # No force if coordinates coincide
        # Collision check
        #if self.check_collision(other):
        if self.check_collision(self.x+self.vx*TIMESTEP, self.y+self.vy*TIMESTEP, other):
            print(f"Collision: {self.name} with {other.name}")
            if self.mass < other.mass:
                self.visible = False
            elif self.mass > other.mass:
                other.visible = False
            else:
                self.visible = False
                other.visible = False
            return 0, 0
        force = np.longdouble(G * self.mass * other.mass / dist)  # Newton's law of gravitation
        theta = np.longdouble(np.arctan2(dy, dx))  # Direction angle of the force
        fx = np.longdouble(np.cos(theta) * force)  # Force in X
        fy = np.longdouble(np.sin(theta) * force)  # Force in Y
        return fx, fy
    
def update_positions(bodies, timestep=TIMESTEP):
    new_coords = []
    for body in bodies:
        total_fx = total_fy = np.longdouble(0)  # Total forces
        # Save previous velocity
        #prev_vx = np.longdouble(body.vx)
        #prev_vy = np.longdouble(body.vy)
        if body.visible == False:
            new_coords.append( (float(body.x), float(body.y)) )
            continue
        for other in bodies:
            if body == other or not other.visible:
                continue  # Do not attract itself
            fx, fy = body.attract(other)  # Gravity from other bodies
            total_fx += fx
            total_fy += fy

        # Update velocity by Newton's law
        body.vx += np.longdouble(total_fx / body.mass * TIMESTEP)
        body.vy += np.longdouble(total_fy / body.mass * TIMESTEP)
        # Update coordinates - take the average between previous and new velocity
        dx = np.longdouble(body.vx * TIMESTEP)
        dy = np.longdouble(body.vy * TIMESTEP)
        new_coords.append( (float(body.x + dx), float(body.y + dy)) )
        #body.x += 0.5 * (prev_vx + body.vx) * TIMESTEP
        #body.y += 0.5 * (prev_vy + body.vy) * TIMESTEP
        
    for ind, body in enumerate(bodies):
        body.x, body.y = new_coords[ind]
        # Save orbit points for visualization
        if len(body.orbit) > 500:
            body.orbit.pop(0)
        body.orbit.append((float(body.x), float(body.y)))

def distance_point_to_line(px, py, x1, y1, x2, y2):
    """Return shortest distance from point P(px,py) to the line segment A(x1,y1)-B(x2,y2).

    All coordinates are in meters. If A and B are the same point, returns distance P-A.
    Uses projection of vector AP onto AB and clamps to segment.
    """
    # Vector AB
    vx = x2 - x1
    vy = y2 - y1
    # If A and B coincide, return distance to A
    denom = vx*vx + vy*vy
    if denom == 0:
        return math.hypot(px - x1, py - y1)
    # Parameter t of projection of P onto AB: t = dot(AP,AB)/|AB|^2
    t = ((px - x1) * vx + (py - y1) * vy) / denom
    # Clamp t to [0,1] to get closest point on segment
    if t < 0:
        cx, cy = x1, y1
    elif t > 1:
        cx, cy = x2, y2
    else:
        cx = x1 + t * vx
        cy = y1 + t * vy
    return math.hypot(px - cx, py - cy)

def min_distance_sat_planet(bodies, satellite, planet_idx, steps=10000, timestep=TIMESTEP):
    # Copy bodies for simulation
    sim_bodies = [copy.deepcopy(b) for b in bodies]
    sim_satellite = copy.deepcopy(satellite)
    sim_bodies.append(sim_satellite)
    min_dist = float('inf')
    min_step = 0
    for step in range(steps):
    # Gravity and movement for planets
        dist_for_sun_prev = math.sqrt(sim_satellite.x**2 + sim_satellite.y**2)
        planet = sim_bodies[0]
        dist_prev = math.sqrt((sim_satellite.x - planet.x)**2 + (sim_satellite.y - planet.y)**2)
        update_positions(sim_bodies, timestep=timestep)
        dist = math.sqrt((sim_satellite.x - planet.x)**2 + (sim_satellite.y - planet.y)**2)
        #print( "step %d   dist to Earth %.0f   ex: %d %d   sat: %d %d" % (step, dist/1000, sim_satellite.x, sim_satellite.y, planet.x, planet.y) )

        if not sim_satellite.visible:
            return float('inf'), -1, -1
        # Calculate distance
        planet = sim_bodies[planet_idx]
        dist = math.sqrt((sim_satellite.x - planet.x)**2 + (sim_satellite.y - planet.y)**2)
        
        dist_for_sun_curr = math.sqrt(sim_satellite.x**2 + sim_satellite.y**2)
        #if step % 25 == 0:
        #    print( "step %d   dist %.0f" % (step, dist) )
        # visualization of satellite trajectory
        #print(globals())
        #print()
        '''
        win = globals().get('win', None)
        win.fill((0, 0, 0))  # Clear screen

        # Event handling (window close)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit(0)
            if event.type == pygame.KEYDOWN:
                return min_dist, min_step * timestep, min_step
                #if event.key == pygame.K_ESCAPE:
                #    run = False

        for body in sim_bodies:
            body.draw(win)  # Draw body
        pygame.display.update()  # Update screen
        '''
        if dist < min_dist:
            min_dist = dist
            min_step = step
        
        #if dist_for_sun_curr > dist_for_sun_prev and dist > planet.orbital_radius * 1.1:
        #    print("  break at step %d with dist %e" % (step, min_dist))
        #    break
    return min_dist, min_step * timestep, min_step

def optimize_satellite_angle(bodies, satellite, 
                             start_idx, dest_idx, 
                             vmin=8000, vmax=12000, vstep=1000, 
                             angle_min=-90, angle_max=90, angle_step=1, 
                             #delay_min=0, delay_max=0, delay_step=0,
                             steps=10000):
    best_angle = None
    best_speed = None
    #best_delay = None 
    best_dist = float('inf')
    best_time = 0
    earth = bodies[start_idx]
    mercury = bodies[dest_idx]
    #dx = mercury.x - earth.x
    #dy = mercury.y - earth.y
    sdist = earth.real_radius + 400*1000
    #base_angle = math.atan2(dy, dx)
    sat_copy = copy.deepcopy(satellite)
    for da in range(int((angle_max - angle_min) / angle_step) + 1):
        for vind in range(int((vmax - vmin) / vstep) + 1):
            #for delay_ind in range(int((delay_max - delay_min) / delay_step) + 1):
            angle_deg = angle_min + da * angle_step
            angle_rad = math.radians(angle_deg)#base_angle + math.radians(angle_deg)
            v0 = vmin + vind * vstep
            #delay = delay_min + delay_ind * delay_step
            sat_copy.x = earth.x+sdist*math.cos(angle_rad)
            sat_copy.y = earth.y+sdist*math.sin(angle_rad)
            sat_copy.vx = v0 * math.cos(angle_rad+math.pi/2)
            sat_copy.vy = v0 * math.sin(angle_rad+math.pi/2)
            #print( "speed vx %.0f vy %.0f " % (sat_copy.vx, sat_copy.vy ) )
            sat_copy.name = f"Sat {angle_deg:.1f}°"
            # Create a copy of bodies and add the satellite
            test_bodies = [copy.deepcopy(b) for b in bodies]
            min_dist, min_time, min_step = min_distance_sat_planet(test_bodies, satellite=sat_copy, planet_idx=dest_idx, steps=steps, timestep=TIMESTEP)
            #results.append((angle_deg, min_dist, min_time))
            print( "angle: %.5f     speed   %.5f    dist: %e   step: %d" % (angle_deg, v0, min_dist, min_step) )
            if min_dist < best_dist:
                best_dist = min_dist
                best_angle = angle_deg
                best_time = min_time
                best_speed = v0
                #best_delay = delay
                #best_step = min_step
    if best_angle is not None:
        print(f"Best angle: {best_angle:.2f}°,   best speed: {best_speed:.3f},    min distance: {best_dist:.2e} m, time: {best_time/86400:.2f} days")
    return best_angle, best_speed, best_dist, best_time#, results

def optimize_satellite_angle_recurent(bodies, satellite, 
                                      start_idx, dest_idx, 
                                      vmin=8000, vmax=12000, vstep = 500, 
                                      angle_min=-180, angle_max=180, angle_step=1,
                                      #delay_min=0, delay_max=0, delay_step=0,
                                      steps=10000):
    for k in range(5):
        print("================================ Iteration %d %.5f %.5f %.5f ===========================" 
            % (k+1, angle_min, angle_max, angle_step))
        (best_angle, best_v, best_dist, best_time) = optimize_satellite_angle(bodies, satellite, 
                                                                      start_idx=start_idx, dest_idx=dest_idx, 
                                                                      vmin=vmin, vmax=vmax, vstep=vstep,
                                                                      #delay_min=delay_min, delay_max=delay_max, delay_step=delay_step,
                                                                      angle_min=angle_min, angle_max=angle_max, angle_step=angle_step)
        if best_angle is None:
            break
        angle_min = best_angle - 2*angle_step
        angle_max = best_angle + 2*angle_step
        angle_step = angle_step / 4
        #delay_min = best_delay - 2*delay_step
        #delay_max = best_delay + 2*delay_step
        #delay_step = delay_step / 4
        vmin = best_v - 2*vstep
        vmax = best_v + 2*vstep
        vstep = vstep / 4
    #print(f"Best angle: {best_angle:.2f}°, min. distance: {best_dist:.2e} m, time: {best_time/86400:.2f} days")
    return best_angle

 # Main function that runs the simulation
def main():
    global SCALE, TIMESTEP
    global CENTER_X, CENTER_Y
    # Create objects: Sun and planets
    '''
    sun = Body(0, 0, 20, YELLOW, 1.98892e30, real_radius=6.9634e8, name="Sun")
    mercury = Body(5.79e10, 0, 6, GRAY, 3.30e23, vy=47870, real_radius=2.4397e6, name="Mercury")
    venus = Body(1.082e11, 0, 8, WHITE, 4.8685e24, vy=35020, real_radius=6.0518e6, name="Venus")
    earth = Body(1.496e11, 0, 10, BLUE, 5.9742e24, vy=29780, real_radius=6.371e6, name="Earth")
    # Moon: orbit radius ~384,400 km, speed ~1022 m/s, mass 7.34767309e22 kg, radius 1.7371e6 m
    #moon = Body(1.496e11 + 1e10, 0, 3, GRAY, 7.34767309e22, vy=29780, vx = 8000, real_radius=1.7371e6, name="Moon")
    mars = Body(2.279e11, 0, 8, RED, 6.39e23, vy=24077, real_radius=3.3895e6, name="Mars")
    jupiter = Body(7.785e11, 0, 16, ORANGE, 1.898e27, vy=13070, real_radius=6.9911e7, name="Jupiter")
    saturn = Body(1.433e12, 0, 12, BROWN, 5.683e26, vy=9680, has_rings=True, real_radius=5.8232e7, name="Saturn")
    '''
    earth = Body(0, 0, 10, BLUE, 5.9726e24, real_radius=6.371e6, name="Earth")
    moon = Body(earth.x+3.844e8, 0, 3, GRAY, 7.34767309e22, vy=1023, vx = 0, real_radius=1.7371e6, name="Moon")
    first_correction = False

    sat_speed = 10846.875
    size = 5
    #satellite = Body(earth.x, earth.y+earth.real_radius-size+1000, 3, WHITE, mass=1e3, vx=vx, vy=vy, real_radius=size, name="Satellite")
    sdist = earth.real_radius + 400*1000
    angle_rad = math.radians(-142.0)
    x = earth.x+sdist*math.cos(angle_rad)
    y = earth.y+sdist*math.sin(angle_rad)
    vx = sat_speed * math.cos(angle_rad+math.pi/2)
    vy = sat_speed * math.sin(angle_rad+math.pi/2)
    satellite = Body(x, y, 3, WHITE, mass=1e3, vx=vx, vy=vy, real_radius=size, name="Satellite")

    #bodies = [sun, mercury, venus, earth, mars, jupiter, saturn]  # List of all bodies
    bodies = [earth, moon, satellite]  # List of all bodies
    '''
    sat_angle = optimize_satellite_angle_recurent(bodies, satellite, 
                                                  start_idx=0, dest_idx=1, 
                                                  vmin=9000, vmax=12000, vstep=200,
                                                  #delay_min=0, delay_max=1000, delay_step=100,
                                                  angle_min=-180, angle_max=180, angle_step=10,
                                                  steps=5000)
    return
    if sat_angle is None:
        print("Could not find optimal angle.")
        return
    #dx = mercury.x - earth.x
    #dy = mercury.y - earth.y
    #base_angle = math.atan2(dy, dx)
    angle_rad = math.radians(sat_angle)
    satellite.vx = sat_speed * math.cos(angle_rad)
    satellite.vy = sat_speed * math.sin(angle_rad)
    '''
    
    run = True
    clock = pygame.time.Clock()
    step = 0
    while run:
        clock.tick(120)  # Limit FPS
        win.fill((0, 0, 0))  # Clear screen

        # Event handling (window close)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    run = False
            if event.type == pygame.K_MINUS or event.type == pygame.K_UNDERSCORE or event.type == pygame.KSCAN_LSHIFT:
                SCALE *= 0.8
            if event.type == pygame.K_PLUS or event.type == pygame.KSCAN_RSHIFT:
                SCALE *= 1.25

        dist = math.sqrt((satellite.x - bodies[1].x)**2 + (satellite.y - bodies[1].y)**2)-satellite.real_radius - bodies[1].real_radius
        speed = math.sqrt(satellite.vx**2 + satellite.vy**2)
        print( "step %d   dist to Moon %.0f    sat: %d %d    Vsat: %d" % (step, dist/1000, satellite.x/1000, satellite.y/1000, speed) )
        # Update positions and velocities of all bodies
        update_positions(bodies, timestep=TIMESTEP)

        if step > 1700:
            TIMESTEP = 10 
        if dist < 130000 and not first_correction:
            koeff = 1900 / speed
            satellite.vx *= koeff
            satellite.vy *= koeff
            first_correction = True
        if bodies[-1].visible:
            CENTER_X, CENTER_Y = (bodies[-1].x+bodies[1].x)//2, (bodies[-1].y+bodies[1].y)//2
            #(self.x-CENTER_X) * SCALE + WIDTH // 2
            #x_km * scale = x_pix
            SCALE_X = WIDTH / (abs(bodies[-1].x - bodies[1].x))
            SCALE_Y = HEIGHT / (abs(bodies[-1].y - bodies[1].y))
            SCALE = min( 1e-4, min(SCALE_X, SCALE_Y)*0.5)
        else:
            CENTER_X, CENTER_Y = 0, 0
            SCALE = 2e-6
        for body in bodies:
            # Save satellite position
            #if idx == len(bodies) - 1:
            #    sat_positions.append([body.x, body.y])
            body.draw(win)  # Draw body

        step += 1
        #if step % 1 == 0:
        #    dist = math.sqrt((satellite.x - sun.x)**2 + (satellite.y - sun.y)**2)
        #    print( "step %d    dist_to_sun %e    %.1f %%   , speed (%.0f / %.0f) " % (step, dist, 100*dist/sun.real_radius, satellite.vx, satellite.vy ) )
        #    if (step == 250):
        #        break
        pygame.display.update()  # Update screen

    # Save satellite trajectory to file
    '''
    with open('sat0.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y'])
        writer.writerows(sat_positions)
    '''
    pygame.quit()  # Quit pygame

# Program entry point
if __name__ == "__main__":
    main()