import cv2

def main():
    # Inicializar as câmeras
    cap_right = cv2.VideoCapture(2)  # Camera direita
    cap_left = cv2.VideoCapture(4)  # Camera direita
    cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
  
    while True:

        # Capturar frames das câmeras
        ret_right, frame_right = cap_right.read()
        ret_left, frame_left = cap_left.read()

        # Exibir as imagens
        cv2.imshow('Camera Direita', frame_right)
        cv2.imshow('Camera Esquerda', frame_left)


        # Verificar se o usuário pressionou a tecla 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar os recursos
    cap_right.release()
    cap_left.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
