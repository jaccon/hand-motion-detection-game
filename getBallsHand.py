import cv2
import pygame
import random
import numpy as np

# Inicializar Pygame
pygame.init()

# Configurações da tela
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Motion Detection Game")

# Configurações das bolinhas
ball_radius = 15
ball_color = (0, 0, 255)
ball_speed = 5

# Lista para armazenar as bolinhas
balls = []

# Variável de pontuação
score = 0

# Fonte para exibir a pontuação
font = pygame.font.SysFont(None, 36)

# Captura de vídeo
capture = cv2.VideoCapture(0)

# Função para criar uma nova bolinha
def create_ball():
    x = random.randint(ball_radius, width - ball_radius)
    y = -ball_radius
    return [x, y]

# Função para desenhar bolinhas e a pontuação
def draw(screen, balls, score, frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.rot90(frame)
    frame = pygame.surfarray.make_surface(frame)
    screen.blit(frame, (0, 0))
    for ball in balls:
        pygame.draw.circle(screen, ball_color, (ball[0], ball[1]), ball_radius)
    score_text = font.render(f"Score: {score}", True, (0, 0, 0))
    screen.blit(score_text, (10, 10))
    pygame.display.flip()

# Leitura do primeiro quadro da câmera
ret, frame1 = capture.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)

# Loop principal do jogo
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Leitura do próximo quadro da câmera
    ret, frame2 = capture.read()
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)
    
    # Cálculo da diferença entre os quadros
    frame_delta = cv2.absdiff(gray1, gray2)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Atualização das bolinhas
    for ball in balls:
        ball[1] += ball_speed
        if ball[1] > height + ball_radius:
            balls.remove(ball)
    
    # Verificação de colisões com detecção de movimento
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        for ball in balls:
            if (x < ball[0] < x + w) and (y < ball[1] < y + h):
                balls.remove(ball)
                score += 1
    
    # Criação de novas bolinhas
    if random.randint(1, 20) == 1:
        balls.append(create_ball())
    
    # Desenho das bolinhas e da pontuação
    draw(screen, balls, score, frame2)
    
    # Atualização do quadro de referência
    gray1 = gray2
    
    # Controle de frames por segundo
    clock.tick(30)

# Encerramento
capture.release()
pygame.quit()
cv2.destroyAllWindows()
