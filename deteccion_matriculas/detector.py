import cv2
import os

modelo_path = os.path.join("modelos", "haarcascade_russian_plate_number.xml")
detector_placas = cv2.CascadeClassifier(modelo_path)

def difuminar_matricula(imagen_path, salida_path):
    imagen = cv2.imread(imagen_path)
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    placas = detector_placas.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in placas:
        region = imagen[y:y+h, x:x+w]
        w_region = w
        h_region = h
        ksize = (max(31, w_region // 3 * 2 | 1), max(31, h_region // 3 * 2 | 1))  # Siempre impar
        region_borrosa = cv2.GaussianBlur(region, ksize, 0)
        imagen[y:y+h, x:x+w] = region_borrosa

    cv2.imwrite(salida_path, imagen)
    return True if len(placas) > 0 else False