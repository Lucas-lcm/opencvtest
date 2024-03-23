import cv2 #Importanto o pacote cv2 para reconhecimento do imagens

carregaFace = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml') #Carregando o algoritmo XML de frontalface
carregaOlho = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml') #Carregando o algoritmo XML de yey

imagemFace = cv2.imread('Fotos pessoas/meneson.jpg') #Lendo a imagem da pasta
imagemFaceCinza = cv2.cvtColor(imagemFace, cv2.COLOR_BGR2GRAY) #Transformando a imagem em escala de cinza

faces = carregaFace.detectMultiScale(imagemFaceCinza)

#Looping for para desenhar um retangulo na face e reconhecer os olhos da face detectada

for (x, y, l, a) in faces:
    faceDetect = cv2.rectangle(imagemFace, (x, y), (x+l, y+a), (0, 255, 0), 2)
    yey = faceDetect[y:y + a, x:x +l]
    yeyCinza = cv2.cvtColor(yey, cv2.COLOR_BGR2GRAY)
    yeyDetect = carregaOlho.detectMultiScale(yeyCinza)

    for (ox, oy, ol, oa) in yeyDetect:
        cv2.rectangle(yey, (ox, oy), (ox + ol, oy + oa), (255, 0, 255), 2)


#Métodos do cv2 para mostrar a a imagem identificada
cv2.imshow('Faces e olhos', imagemFace)
cv2.waitKey()
