from PIL import Image
from nn import StyleTransfer
from timeit import default_timer as timer

import numpy as np
import pygame
import sys


SIZE = WIDTH, HEIGHT = 300, 300
STYLE_IMAGE_PATH = "../resources/imagen2.jpg"

st = StyleTransfer()
btl = st.bottleneck(STYLE_IMAGE_PATH)
# cnt = st.content('resources/hielo.jpg')

pygame.init()
screen = pygame.display.set_mode(SIZE)
pygame.display.set_caption('Style Transfer Benchmark')

total = 0

for i in range(0, 30):
    start = timer()
    cnt = st.preprocess_image(st.random_img(), 384)
    img = st.transfer_style(btl, cnt) * 255
    surface = pygame.surfarray.make_surface(img)
    screen.blit(surface, (0, 0))
    pygame.display.update()
    end = timer()
    print(f'Loop #{i}: {end - start} segundos')
    total += end - start

print(f'Tiempo promedio por frame: {total / 30}')

# run = True
# while run:
#    for event in pygame.event.get():
#        if event.type == pygame.QUIT:
#            pygame.quit
#            break
#        if event.type == pygame.KEYDOWN:
#            if event.key == pygame.K_SPACE:
#                print('Apretaste space')
#            
#            elif event.key == pygame.K_ESCAPE:
#                print('Escape')
#                pygame.quit()
#                break

pygame.quit()

print('FIN')
