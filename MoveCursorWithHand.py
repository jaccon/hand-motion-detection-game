import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Inicialização do MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Captura de vídeo
capture = cv2.VideoCapture(0)

# Configurações para a detecção de mãos com o MediaPipe
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5) as hands:

    screen_width, screen_height = pyautogui.size()

    while True:
        # Leitura do próximo quadro da câmera
        ret, frame = capture.read()
        if not ret:
            break
        
        # Inverter a imagem horizontalmente
        frame = cv2.flip(frame, 1)

        # Converter o quadro para RGB para o MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detectar mãos no quadro
        results = hands.process(rgb_frame)

        # Verificar se foram detectadas mãos
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Desenhar pontos das mãos no frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2))

                # Obter posição do indicador (ponta do dedo)
                index_finger_pos = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                finger_x = int(index_finger_pos.x * frame.shape[1])
                finger_y = int(index_finger_pos.y * frame.shape[0])

                # Mapear posição do dedo para posição do cursor na tela
                cursor_x = int(np.interp(finger_x, [0, frame.shape[1]], [0, screen_width]))
                cursor_y = int(np.interp(finger_y, [0, frame.shape[0]], [0, screen_height]))

                # Movimento suave do cursor
                current_mouse_x, current_mouse_y = pyautogui.position()
                smooth_cursor_x = current_mouse_x + (cursor_x - current_mouse_x) / 5
                smooth_cursor_y = current_mouse_y + (cursor_y - current_mouse_y) / 5

                # Mover o cursor
                pyautogui.moveTo(smooth_cursor_x, smooth_cursor_y)

        # Exibir o frame com a detecção de mãos e controle do cursor
        cv2.imshow('Hand Tracking and Cursor Control', frame)

        # Sair do loop se a tecla 'q' for pressionada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberar recursos
capture.release()
cv2.destroyAllWindows()
