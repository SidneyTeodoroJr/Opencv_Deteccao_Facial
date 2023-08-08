# importar opencv
import cv2

# Lista de caminhos para as imagens que serão processadas
image_paths = ['img/comemoracao.png', 'img/mulher_loira.png', 'img/mulher_na_janela.jpg', 'img/crianca_menina.jpg', 'img/irmaos_crianca.jpg']
# Caminho para o arquivo XML que contém o modelo de cascata de detecção facial
cascade_path = 'haarcascade_frontalface_default.xml'


clf = cv2.CascadeClassifier(cascade_path) # Cria um classificador de cascata de detecção facial
for image_path in image_paths:
    img = cv2.imread(image_path)  # Lê a imagem
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converte a imagem para escala de cinza (preto e branco)
    faces = clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)  # Detecta faces na imagem utilizando o classificador

    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)  # Desenha um retângulo em volta da face detectada na imagem original

    # Exibe a imagem com as faces detectadas
    cv2.imshow('image', img)
    cv2.waitKey(0)

# Fecha todas as janelas abertas
cv2.destroyAllWindows()