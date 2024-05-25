import cv2

def capture_photo(camera_index, file_name):
    # Inicializar a captura de vídeo da câmera
    cap = cv2.VideoCapture(camera_index)
    
    # Verificar se a câmera foi aberta corretamente
    if not cap.isOpened():
        print(f"Não foi possível abrir a câmera {camera_index}.")
        return
    
    # Capturar um frame da câmera
    ret, frame = cap.read()
    
    # Verificar se o frame foi capturado corretamente
    if not ret:
        print(f"Falha ao capturar o frame da câmera {camera_index}.")
        cap.release()
        return
    
    # Salvar o frame como uma imagem
    cv2.imwrite(file_name, frame)
    print(f"Foto salva como {file_name}")

    # Exibir a foto
    cv2.imshow(file_name, frame)

def main():
    # Pedir ao usuário para digitar o valor de x
    x = 50

    # Capturar fotos das duas webcams
    capture_photo(4, f"left_eye_{x}cm.jpg")
    capture_photo(2, f"right_eye_{x}cm.jpg")

if __name__ == "__main__":
    main()
