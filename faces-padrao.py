import cv2 #Importanto o pacote cv2 para reconhecimento do imagens

carregaAlgoritmo = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml') #Carregando o algoritmo XML de frontalface

imagem = cv2.imread('Fotos pessoas/Pessoas-felizes-1024x682.jpg') #Lendo a imagem da pasta
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) #Transformando a imagem em escala de cinza

faces = carregaAlgoritmo.detectMultiScale(imagemCinza, scaleFactor=1.09, minNeighbors=5, minSize=(35,35)) #Analisando faces adicionando o scaleFactor, minNeighbors e minSize

print(faces) #Printando resultado (Matriz de posição onde estão as faces (eixo x, eixo y, largura e altura)

#Looping for para desenhar um retangulo na face

for(x, y, l, a) in faces:
    cv2.rectangle(imagem, (x, y), (x+l, y+a), (0, 255, 0), 2)

#Métodos do cv2 para mostrar a a imagem identificada
cv2.imshow('Faces', imagem)
cv2.waitKey()

