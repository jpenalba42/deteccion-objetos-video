from ultralytics import YOLO
import cv2
import os

modelo_path = os.path.join("modelos", "license-plate-finetune-v1l.pt")
modelo = YOLO(modelo_path)

def difuminar_matricula_yolo(imagen_path, salida_path):
    imagen = cv2.imread(imagen_path)
    resultados = modelo(imagen_path)
    hay_placa = False

    for r in resultados:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            region = imagen[y1:y2, x1:x2]
            region_borrosa = cv2.GaussianBlur(region, (23, 23), 30)
            imagen[y1:y2, x1:x2] = region_borrosa
            hay_placa = True

    cv2.imwrite(salida_path, imagen)
    return hay_placa