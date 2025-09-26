import pygame  # Библиотека для визуализации
import math    # Математические функции

###########################################################
# Константы
###########################################################
WIDTH, HEIGHT = 2000, 800  # Размер окна
G = 6.67430e-11            # Гравитационная постоянная
SCALE = 6e-10               # Масштаб для визуализации (уменьшает расстояния)
TIMESTEP = 7200            # Временной шаг (1 час за кадр)


# Цвета для планет
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
BLUE = (100, 149, 237)
GRAY = (128, 128, 128)
RED = (188, 39, 50)


# Класс, описывающий небесное тело (планету или звезду)
class Body:
    def __init__(self, x, y, radius, color, mass, vx=0, vy=0, has_rings=False):
        self.x = x              # Координата X (м)
        self.y = y              # Координата Y (м)
        self.radius = radius    # Радиус для отрисовки (пиксели)
        self.color = color      # Цвет
        self.mass = mass        # Масса (кг)
        self.vx = vx            # Скорость по X (м/с)
        self.vy = vy            # Скорость по Y (м/с)
        self.orbit = []         # Список точек орбиты для визуализации
        self.has_rings = has_rings  # Есть ли кольца

    def draw(self, win):
        """Отрисовка тела и его орбиты на экране"""
        x = self.x * SCALE + WIDTH // 2
        y = self.y * SCALE + HEIGHT // 2
        # Рисуем кольца для Сатурна
        if self.has_rings:
            ring_color = (200, 200, 180)
            # Наружный эллипс (кольцо)
            pygame.draw.ellipse(win, ring_color, [int(x - self.radius*2.5), int(y - self.radius*1.2), int(self.radius*5), int(self.radius*2.4)], 2)
            # Внутренний эллипс (тонкая часть кольца)
            pygame.draw.ellipse(win, ring_color, [int(x - self.radius*1.5), int(y - self.radius*0.7), int(self.radius*3), int(self.radius*1.4)], 1)
        pygame.draw.circle(win, self.color, (int(x), int(y)), self.radius)
        # Рисуем линию орбиты с разной толщиной
        n = len(self.orbit)
        if n > 2:
            points = [(ox * SCALE + WIDTH // 2, oy * SCALE + HEIGHT // 2) for ox, oy in self.orbit]
            one_third = n // 3
            two_third = 2 * n // 3
            # Первая треть (старые точки)
            if one_third > 1:
                pygame.draw.lines(win, self.color, False, points[:one_third], 1)
            # Средняя треть
            if two_third - one_third > 1:
                pygame.draw.lines(win, self.color, False, points[one_third:two_third], 2)
            # Последняя треть (самые свежие точки)
            if n - two_third > 1:
                pygame.draw.lines(win, self.color, False, points[two_third:], 3)

    def attract(self, other):
        """Вычисляет гравитационное притяжение между телами"""
        dx = other.x - self.x
        dy = other.y - self.y
        dist = math.sqrt(dx**2 + dy**2)  # Расстояние между телами
        if dist == 0:
            return 0, 0  # Нет силы, если координаты совпадают
        force = G * self.mass * other.mass / dist**2  # Закон всемирного тяготения
        theta = math.atan2(dy, dx)  # Угол направления силы
        fx = math.cos(theta) * force  # Сила по X
        fy = math.sin(theta) * force  # Сила по Y
        return fx, fy


# Главная функция, запускающая симуляцию
def main():
    def min_distance_sat_mercury(bodies, satellite_idx, mercury_idx, steps=10000, timestep=TIMESTEP):
        # Копируем тела для симуляции
        import copy
        sim_bodies = [copy.deepcopy(b) for b in bodies]
        min_dist = float('inf')
        min_step = 0
        for step in range(steps):
            # Гравитация и движение
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
            for body in sim_bodies:
                body.x += body.vx * timestep
                body.y += body.vy * timestep
            # Считаем расстояние
            sat = sim_bodies[satellite_idx]
            merc = sim_bodies[mercury_idx]
            dist = math.sqrt((sat.x - merc.x)**2 + (sat.y - merc.y)**2)
            if dist < min_dist:
                min_dist = dist
                min_step = step
        return min_dist, min_step * timestep

    pygame.init()  # Инициализация pygame
    win = pygame.display.set_mode((WIDTH, HEIGHT))  # Создание окна
    pygame.display.set_caption("Солнечная система")

    # Создание объектов: Солнце и планеты
    sun = Body(0, 0, 20, YELLOW, 1.98892e30)
    earth = Body(1.496e11, 0, 10, BLUE, 5.9742e24, vy=29780)      # Земля
    mars = Body(2.279e11, 0, 8, RED, 6.39e23, vy=24077)          # Марс
    mercury = Body(5.79e10, 0, 6, GRAY, 3.30e23, vy=47870)        # Меркурий
    venus = Body(1.082e11, 0, 8, WHITE, 4.8685e24, vy=35020)     # Венера
    jupiter = Body(7.785e11, 0, 16, (255, 165, 0), 1.898e27, vy=13070)   # Юпитер
    saturn = Body(1.433e12, 0, 12, (210, 180, 140), 5.683e26, vy=9680, has_rings=True)   # Сатурн с кольцами

    # Новая позиция Меркурия через время встречи
    meet_time = 1.344e7  # секунд
    new_mercury_x = mercury.x
    new_mercury_y = mercury.y + 47870 * meet_time
    # Стартовая точка: Земля
    start_x = earth.x
    start_y = earth.y
    # Вектор направления
    dx = new_mercury_x - start_x
    dy = new_mercury_y - start_y
    dist = math.sqrt(dx**2 + dy**2)
    dir_x = dx / dist
    dir_y = dy / dist
    v0 = 8e3
    vx = dir_x * v0
    vy = dir_y * v0
    satellite = Body(start_x, start_y, 4, WHITE, 1e3, vx=vx, vy=vy)

    bodies = [sun, mercury, venus, earth, mars, jupiter, saturn, satellite]  # Список всех тел
    # Для записи траектории спутника
    import csv
    sat_positions = []

    # Вызываем функцию для оценки минимального расстояния
    min_dist, min_time = min_distance_sat_mercury(bodies, satellite_idx=len(bodies)-1, mercury_idx=3)
    print(f"Минимальное расстояние спутник-Меркурий: {min_dist:.2e} м, время: {min_time/86400:.2f} дней")
    run = True
    clock = pygame.time.Clock()
    while run:
        clock.tick(120)  # Ограничение FPS
        win.fill((0, 0, 0))  # Очистка экрана

        # Обработка событий (закрытие окна)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    run = False

        # Обновление положения и скорости всех тел
        for idx, body in enumerate(bodies):
            total_fx = total_fy = 0  # Суммарные силы
            for other in bodies:
                if body == other:
                    continue  # Не притягиваем сами себя
                fx, fy = body.attract(other)  # Гравитация от других тел
                total_fx += fx
                total_fy += fy

            # Обновление скорости по закону Ньютона
            body.vx += total_fx / body.mass * TIMESTEP
            body.vy += total_fy / body.mass * TIMESTEP
            # Обновление координат
            body.x += body.vx * TIMESTEP
            body.y += body.vy * TIMESTEP
            # Сохраняем точки орбиты для визуализации
            if len(body.orbit) > 500:
                body.orbit.pop(0)
            body.orbit.append((body.x, body.y))
            # Сохраняем позицию спутника
            if idx == len(bodies) - 1:
                sat_positions.append([body.x, body.y])
            body.draw(win)  # Рисуем тело

        pygame.display.update()  # Обновление экрана

    # Сохраняем траекторию спутника в файл
    with open('sat0.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y'])
        writer.writerows(sat_positions)
    pygame.quit()  # Завершение работы pygame

# Запуск программы
if __name__ == "__main__":
    main()