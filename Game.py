import pygame
import random
import numpy as np
import FrameOfReference
import Vision

class Rect:
  def __init__(self, x, y, w, h, col):
    self.x = x
    self.y = y
    self.w = w
    self.h = h
    self.col = col
    self.x1, self.y1 = x, y
    self.x2, self.y2 = x + w, y
    self.x3, self.y3 = x + w, y + h
    self.x4, self.y4 = x, y + h
    self.dop_col = col
    self.draw = True

min_gap = 0
max_gap = 1
max_vert_gap = 0# 1.5
max_height = 0.5
min_length = 1.5
max_length = 5

back_col = (255, 255, 255)
player_col = 650
platform_col = 475

def generate_platforms(curr_x, curr_y, final_x, platforms):
  while curr_x < final_x:
    length = random.random() * (max_length - min_length) + min_length
    height = max_height
    platforms.append(Rect(curr_x, curr_y - height, length, height, platform_col))
    curr_x += (length + random.random() * max_gap - min_gap) + min_gap
    curr_y += (random.random() * 2 - 1) * max_vert_gap
  return curr_x, curr_y

draw_scale = 50

nat_m = 1
nat_s = 1

EPS = 0.001

def get_observation(ref_frame, x, y):
  x, y = x * nat_m, y * nat_m
  t = - x * ref_frame.velocity[0] - y * ref_frame.velocity[1]
  return np.array([t, x, y])

def from_observation(txy):
  t, x, y = txy
  assert(abs(t) < EPS)
  return x / nat_m, y / nat_m

def convert_speed(dx, dy):
  return np.array([nat_m * dx / nat_s, nat_m * dy / nat_s])

def draw_transform_point(x, y):
  return (400 + x * draw_scale, 300 - y * draw_scale)

def reset_rect(r):
  r.x1, r.y1 = r.x - x, r.y - y
  r.x2, r.y2 = r.x + r.w - x, r.y - y
  r.x3, r.y3 = r.x + r.w - x, r.y + r.h - y
  r.x4, r.y4 = r.x - x, r.y + r.h - y

def lorentz_rect(ref_frame, r):
  p1 = get_observation(ref_frame, r.x1, r.y1)
  p2 = get_observation(ref_frame, r.x2, r.y2)
  p3 = get_observation(ref_frame, r.x3, r.y3)
  p4 = get_observation(ref_frame, r.x4, r.y4)

  p1, p2, p3, p4 = ref_frame.inverse_transform_polygon([p1, p2, p3, p4])

  r.x1, r.y1 = from_observation(p1)
  r.x2, r.y2 = from_observation(p2)
  r.x3, r.y3 = from_observation(p3)
  r.x4, r.y4 = from_observation(p4)

  avg_x = (r.x1 + r.x2 + r.x3 + r.x4) / 4
  avg_y = (r.y1 + r.y2 + r.y3 + r.y4) / 4

  # r.dop_col = ref_frame.doppler_shift(r.col, np.array([0, avg_x, avg_y]))

def translate_rect(x, y, r):
  r.x1, r.y1 = r.x1 - x, r.y1 - y
  r.x2, r.y2 = r.x2 - x, r.y2 - y
  r.x3, r.y3 = r.x3 - x, r.y3 - y
  r.x4, r.y4 = r.x4 - x, r.y4 - y

def draw_rect(disp, r):
  x1, y1 = draw_transform_point(r.x1, r.y1)
  x2, y2 = draw_transform_point(r.x2, r.y2)
  x3, y3 = draw_transform_point(r.x3, r.y3)
  x4, y4 = draw_transform_point(r.x4, r.y4)

  if x1 > 800 and x2 > 800 and x3 > 800 and x4 > 800:
    return
  if x1 < 0 and x2 < 0 and x3 < 0 and x4 < 0:
    return
  if y1 > 800 and y2 > 800 and y3 > 800 and y4 > 800:
    return
  if y1 < 0 and y2 < 0 and y3 < 0 and y4 < 0:
    return
  red, green, blue = Vision.wavelength_to_rgb(r.dop_col)
  pygame.draw.polygon(disp, (red, green, blue), [(x1, y1), (x2, y2), (x3, y3), (x4, y4)])

def move_point(x, y, dx, dy):
  return (x + dx, y + dy)

player_width = 1
player_height = 1
player_mass = 1

if __name__ == "__main__":
  platforms = []
  generate_platforms(-player_width / 2, -player_height / 2, 100, platforms)
  print(len(platforms))

  pygame.init()
  pygame.display.set_caption('Relativistic Runner')
  game_display = pygame.display.set_mode((800, 600))
  game_clock = pygame.time.Clock()

  m = 1
  x, y = 0, 0
  px, py = 0, 0
  fx, fy = 0, 0
  g = -0.01
  dt = 0.7

  player_rect = Rect(-player_width / 2, -player_height / 2, player_width, player_height, player_col)

  grounded = True

  ref_frame = FrameOfReference.LabFrame(np.zeros(2))

  stop = False
  while not stop:

    game_display.fill(back_col)
    draw_rect(game_display, player_rect)
    for r in platforms:
      if r.draw:
        draw_rect(game_display, r)
    pygame.display.update()

    game_clock.tick(60)

    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        stop = True
      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_RIGHT:
          fx = 0.01
        elif event.key == pygame.K_LEFT:
          fx = -0.01
        elif event.key == pygame.K_SPACE:
          if grounded:
            py = 0.2
      elif event.type == pygame.KEYUP:
        if event.key == pygame.K_RIGHT or event.key == pygame.K_LEFT:
          fx = 0

    fy = g * player_mass
    f_3vec = np.array([0, fx, fy])
    f_3vec_prime = ref_frame.transform(f_3vec)
    dt_prime = dt * ref_frame.gamma
    m_prime = ref_frame.get_mass(player_mass)

    px, py = move_point(px, py, f_3vec_prime[1] * dt / m_prime, f_3vec_prime[2] * dt / m_prime)
    print(np.linalg.norm(ref_frame.velocity))
    x, y = move_point(x, y, px * dt, py * dt)

    ref_frame.update(convert_speed(px / m_prime, py / m_prime))

    grounded = False

    for r in platforms:
      reset_rect(r)
      translate_rect(x, y, r)

      x1, y1 = r.x1, r.y1
      x2, y2 = r.x3, r.y3
      
      horz_in = x1 < player_width / 2 and x2 > -player_width / 2
      vert_in = y1 < player_height / 2 and y2 > -player_height / 2
      is_in = horz_in and vert_in
      if is_in:
        if px > 0:
          horz_move = x1 - player_width / 2
        else:
          horz_move = x2 + player_width / 2
        if py > 0:
          vert_move = y1 - player_height / 2
        else:
          vert_move = y2 + player_height / 2
        
        if abs(horz_move) < abs(vert_move):
          x += horz_move
          px = 0
        else:
          if (vert_move > 0):
            grounded = True
          y += vert_move
          py = 0

  pygame.quit()
  quit()
