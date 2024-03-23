import cv2 #Importanto o pacote cv2 para reconhecimento do imagens

webcam = cv2.VideoCapture(1) #Definindo uma variável que irá receber a caputra da minha webcam
classificadorVideoFace  = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')  #Importando o XML de reconhecimento de face

#looping que irá realizar a leitura da face aparecendo na webcam e colcoar um retangulo na face detectada.

while True:
    camera, frame = webcam.read() #Definindo as variaveis que serao utilizadas

    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Tranformando a imagem em escala de cinza
    detecta = classificadorVideoFace.detectMultiScale(cinza, scaleFactor=1.09, minNeighbors=5, minSize=(35,35)) #Detectando a face

#looping for que irá desenhar o retangulo na face destectada

    for(x, y, l, a) in detecta:
        cv2.rectangle(frame, (x, y), (x + l, y + a), (255, 0, 0), 2)


    cv2.imshow('Imagem WebCamera', frame) #Exibindo o resultado

#IF que encerra a captura

    if cv2.waitKey(1) == ord('q'):
        break

#Métodos que resetam os dados setados
webcam.release()
cv2.destroyAllWindows()
