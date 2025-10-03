import pygame  # Library for visualization
import math    # Mathematical functions
import csv  # For saving the satellite trajectory
import math
import copy


###########################################################
# Constants
###########################################################
WIDTH, HEIGHT = 2000, 1000  # Window size
G = 6.67430e-11            # Gravitational constant
SCALE = 6e-10               # Scale for visualization (reduces distances)
TIMESTEP = 7200            # Time step (1 hour per frame)



# Colors for planets
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
BLUE = (100, 149, 237)
GRAY = (128, 128, 128)
RED = (188, 39, 50)


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
    
    def __init__(self, x, y, radius, color, mass, vx=0, vy=0, has_rings=False, real_radius=None, visible=True):
        self.x = x              # X coordinate (m)
        self.y = y              # Y coordinate (m)
        self.radius = radius    # Radius for drawing (pixels)
        self.color = color      # Color
        self.mass = mass        # Mass (kg)
        self.vx = vx            # Velocity in X (m/s)
        self.vy = vy            # Velocity in Y (m/s)
        self.orbit = []         # List of orbit points for visualization
        self.has_rings = has_rings  # Has rings
        self.real_radius = real_radius  # Real radius (m)
        self.visible = visible  # Body visibility

    def draw(self, win):
        """Draw the body and its orbit on the screen"""
        if not self.visible:
            return
        x = self.x * SCALE + WIDTH // 2
        y = self.y * SCALE + HEIGHT // 2
        # Draw rings for Saturn
        if self.has_rings:
            ring_color = (200, 200, 180)
            # Outer ellipse (ring)
            pygame.draw.ellipse(win, ring_color, [int(x - self.radius*2.5), int(y - self.radius*1.2), int(self.radius*5), int(self.radius*2.4)], 2)
            # Inner ellipse (thin part of the ring)
            pygame.draw.ellipse(win, ring_color, [int(x - self.radius*1.5), int(y - self.radius*0.7), int(self.radius*3), int(self.radius*1.4)], 1)
        pygame.draw.circle(win, self.color, (int(x), int(y)), self.radius)
        # Draw orbit line with varying thickness
        n = len(self.orbit)
        if n > 2:
            points = [(ox * SCALE + WIDTH // 2, oy * SCALE + HEIGHT // 2) for ox, oy in self.orbit]
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
        dx = other.x - self.x
        dy = other.y - self.y
        dist = math.sqrt(dx**2 + dy**2)  # Distance between bodies
        # Collision check
        if self.check_collision(other):
            print(f"Collision: {self} with {other}")
            if self.mass < other.mass:
                self.visible = False
            elif self.mass > other.mass:
                other.visible = False
            else:
                self.visible = False
                other.visible = False
            return 0, 0
        if dist == 0:
            return 0, 0  # No force if coordinates coincide
        force = G * self.mass * other.mass / dist**2  # Newton's law of gravitation
        theta = math.atan2(dy, dx)  # Direction angle of the force
        fx = math.cos(theta) * force  # Force in X
        fy = math.sin(theta) * force  # Force in Y
        return fx, fy
    
def update_positions(bodies, timestep=TIMESTEP):
    for body in bodies:
        total_fx = total_fy = 0  # Total forces
        # Save previous velocity
        prev_vx = body.vx
        prev_vy = body.vy
        if body.visible == False:
            continue
        for other in bodies:
            if body == other or not other.visible:
                continue  # Do not attract itself
            fx, fy = body.attract(other)  # Gravity from other bodies
            total_fx += fx
            total_fy += fy

        # Update velocity by Newton's law
        body.vx += total_fx / body.mass * TIMESTEP
        body.vy += total_fy / body.mass * TIMESTEP
        # Update coordinates - take the average between previous and new velocity
        body.x += 0.5 * (prev_vx + body.vx) * TIMESTEP
        body.y += 0.5 * (prev_vy + body.vy) * TIMESTEP
        # Save orbit points for visualization
        if len(body.orbit) > 500:
            body.orbit.pop(0)
        body.orbit.append((body.x, body.y))

def min_distance_sat_planet(bodies, satellite, planet_idx, steps=10000, timestep=TIMESTEP):
    # Copy bodies for simulation
    sim_bodies = [copy.deepcopy(b) for b in bodies]
    sim_satellite = copy.deepcopy(satellite)
    sim_bodies.append(sim_satellite)
    min_dist = float('inf')
    min_step = 0
    for step in range(steps):
    # Gravity and movement for planets
        for i, body in enumerate(sim_bodies):
            total_fx = total_fy = 0
            for j, other in enumerate(sim_bodies):
                if i == j:
                    continue
                fx, fy = body.attract(other)
                total_fx += fx
                total_fy += fy
            body.vx += total_fx / body.mass * timestep
            body.vy += total_fy / body.mass * timestep
    # Calculate distance
        planet = sim_bodies[planet_idx]
        dist = math.sqrt((sim_satellite.x - planet.x)**2 + (sim_satellite.y - planet.y)**2)
        #if step % 25 == 0:
        #    print( "step %d   dist %.0f" % (step, dist) )
        if dist < min_dist:
            min_dist = dist
            min_step = step
        if dist > min_dist:
            #print("  break at step %d with dist %e" % (step, min_dist))
            break
    return min_dist, min_step * timestep, min_step

    best_angle = None
    best_dist = float('inf')
    best_time = 0
    #best_step = 0
    results = []
    earth = bodies[start_idx]
    mercury = bodies[dest_idx]
    timestep = TIMESTEP
    dx = mercury.x - earth.x
    dy = mercury.y - earth.y
    base_angle = math.atan2(dy, dx)
    for da in range(int((angle_max - angle_min) / angle_step) + 1):
        angle_deg = angle_min + da * angle_step
        angle_rad = base_angle + math.radians(angle_deg)
        vx = v0 * math.cos(angle_rad)
        vy = v0 * math.sin(angle_rad)
    # Create a copy of bodies and add the satellite
        test_bodies = [copy.deepcopy(b) for b in bodies]
        satellite = Body(earth.x, earth.y, 4, WHITE, 1e3, vx=vx, vy=vy)
        test_bodies.append(satellite)
        min_dist, min_time, min_step = min_distance_sat_planet(test_bodies, satellite=satellite, planet_idx=dest_idx, steps=steps, timestep=timestep)
        #results.append((angle_deg, min_dist, min_time))
        print( "angle: %.5f     dist: %e   step: %d" % (angle_deg, min_dist, min_step) )
        if min_dist < best_dist:
            best_dist = min_dist
            best_angle = angle_deg
            best_time = min_time
            #best_step = min_step
    print(f"Best angle: {best_angle:.2f}°, min. distance: {best_dist:.2e} m, time: {best_time/86400:.2f} days")
    return best_angle, best_dist, best_time#, results

    for k in range(5):
        print("================================ Iteration %d %.5f %.5f %.5f ===========================" 
            % (k+1, angle_min, angle_max, angle_step))
        (best_angle, best_dist, best_time) = optimize_satellite_angle(bodies, start_idx=start_idx, dest_idx=dest_idx, v0=v0, 
                                                angle_min=angle_min, angle_max=angle_max, angle_step=angle_step)
        angle_min = best_angle - 2*angle_step
        angle_max = best_angle + 2*angle_step
        angle_step = angle_step / 4
    #print(f"Best angle: {best_angle:.2f}°, min. distance: {best_dist:.2e} m, time: {best_time/86400:.2f} days")

 # Main function that runs the simulation
def main():
    pygame.init()  # Initialize pygame
    win = pygame.display.set_mode((WIDTH, HEIGHT))  # Create window
    pygame.display.set_caption("Solar System")

    # Create objects: Sun and planets
    sun = Body(0, 0, 20, YELLOW, 1.98892e30, real_radius=6.9634e8)
    earth = Body(1.496e11, 0, 10, BLUE, 5.9742e24, vy=29780, real_radius=6.371e6)      # Earth
    # Moon: orbit radius ~384,400 km, speed ~1022 m/s, mass 7.34767309e22 kg, radius 1.7371e6 m
    moon = Body(1.496e11 + 1e10, 0, 3, GRAY, 7.34767309e22, vy=29780, vx = 8000, real_radius=1.7371e6)
    mars = Body(2.279e11, 0, 8, RED, 6.39e23, vy=24077, real_radius=3.3895e6)          # Mars
    mercury = Body(5.79e10, 0, 6, GRAY, 3.30e23, vy=47870, real_radius=2.4397e6)        # Mercury
    venus = Body(1.082e11, 0, 8, WHITE, 4.8685e24, vy=35020, real_radius=6.0518e6)     # Venus
    jupiter = Body(7.785e11, 0, 16, (255, 165, 0), 1.898e27, vy=13070, real_radius=6.9911e7)   # Jupiter
    saturn = Body(1.433e12, 0, 12, (210, 180, 140), 5.683e26, vy=9680, has_rings=True, real_radius=5.8232e7)   # Saturn with rings

    ''' Launching a satellite from Earth towards Mercury
    # New position of Mercury after meeting time
    #meet_time = 1.344e7  # seconds
    #new_mercury_x = mercury.x
    #new_mercury_y = mercury.y + 47870 * meet_time
    # Starting point: Earth
    start_x = earth.x
    start_y = earth.y
    # Direction vector
    dx = mercury.x - start_x
    dy = mercury.y - start_y
    dist = math.sqrt(dx**2 + dy**2)
    dir_x = dx / dist
    dir_y = dy / dist
    v0 = 8e3
    vx = dir_x * v0
    vy = dir_y * v0
    satellite = Body(start_x, start_y, 4, WHITE, 1e3, vx=vx, vy=vy, real_radius=1)  # Satellite
    '''
    dx = mercury.x - earth.x
    dy = mercury.y - earth.y
    base_angle = math.atan2(dy, dx)

    v0 = 8000
    angle_deg = 0
    angle_rad = base_angle + math.radians(angle_deg)
    vx = v0 * math.cos(angle_rad)
    vy = v0 * math.sin(angle_rad)
    size = 5
    satellite = Body(earth.x, earth.y-earth.real_radius-size-100, 4, WHITE, 1e3, vx=vx, vy=vy, real_radius=size)
    
    bodies = [sun, mercury, venus, earth, moon, mars, jupiter, saturn, satellite]  # List of all bodies

    # Example usage: find optimal angle for approach to Mercury
    #angle_deg, min_dist, min_time, final_angle = find_optimal_satellite_angle(
    #    bodies, earth, target_planet_idx=1, v0=8000, max_iter=10, angle_eps=0.01, steps=10000, timestep=7200)
    #print(f"\nOptimal launch angle: {angle_deg:.3f}°, minimum distance: {min_dist:.2e} m, time: {min_time/86400:.2f} days, final angle: {final_angle:.3f}°\n")

    # For saving the satellite trajectory
    #sat_positions = []
   #optimize_satellite_angle(bodies, start_idx=3, dest_idx=1, v0=8000, angle_min=-90, angle_max=90, angle_step=1, steps=20000)
    #optimize_satellite_angle_recurent(bodies, start_idx=3, dest_idx=1, v0=8000, angle_min=-90, angle_max=90, angle_step=2)

    # Call function to estimate minimum distance
    #min_dist, min_time = min_distance_sat_mercury(bodies, satellite_idx=len(bodies)-1, mercury_idx=3)
    #print(f"Minimum distance satellite-Mercury: {min_dist:.2e} m, time: {min_time/86400:.2f} days")
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

        # Update positions and velocities of all bodies
        update_positions(bodies, timestep=TIMESTEP)

        for body in bodies:
            # Save satellite position
            #if idx == len(bodies) - 1:
            #    sat_positions.append([body.x, body.y])
            body.draw(win)  # Draw body

        step += 1
        if step % 1 == 0:
            dist = math.sqrt((satellite.x - sun.x)**2 + (satellite.y - sun.y)**2)
            print( "step %d    dist_to_sun %e    %.1f %%   , speed (%.0f / %.0f) " % (step, dist, 100*dist/sun.real_radius, satellite.vx, satellite.vy ) )
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