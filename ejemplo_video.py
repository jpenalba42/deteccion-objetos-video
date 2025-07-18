#!/usr/bin/env python3
"""
Ejemplo de uso del detector de matrículas en video
"""

from deteccion_matriculas.detector_video_yolo import DetectorVideoYOLO
import os

def ejemplo_basico():
    """Ejemplo básico de detección en video"""
    
    # Crear detector (usa el modelo por defecto)
    detector = DetectorVideoYOLO()
    
    # Ejemplo 1: Procesar un video y guardarlo con detecciones
    video_entrada = "input3.mp4"  # Cambia por tu video
    video_salida = "video_con_detecciones.mp4"
    
    if os.path.exists(video_entrada):
        print("Procesando video con detecciones visibles...")
        detector.detectar_matriculas_video(
            video_path=video_entrada,
            salida_path=video_salida,
            mostrar_video=True,
            difuminar=False,  # Mostrar rectángulos, no difuminar
            confianza=0.5
        )
    
    # Ejemplo 2: Procesar un video difuminando las matrículas
    video_salida_difuminado = "output3.mp4"
    
    if os.path.exists(video_entrada):
        print("Procesando video con matrículas difuminadas...")
        detector.detectar_matriculas_video(
            video_path=video_entrada,
            salida_path=video_salida_difuminado,
            mostrar_video=True,
            difuminar=True,  # Difuminar matrículas
            confianza=0.5
        )

def ejemplo_webcam():
    """Ejemplo de detección en tiempo real con webcam"""
    
    detector = DetectorVideoYOLO()
    
    print("Iniciando detección en webcam...")
    print("Presiona 'q' para salir")
    
    # Detección en tiempo real
    detector.detectar_webcam(
        camara_id=0,  # Cámara por defecto
        difuminar=False,  # Cambiar a True para difuminar
        confianza=0.5
    )

if __name__ == "__main__":
    print("=== Detector de Matrículas en Video ===")
    print("1. Ejemplo básico (requiere video_ejemplo.mp4)")
    print("2. Ejemplo con webcam")
    
    opcion = input("Selecciona una opción (1 o 2): ")
    
    if opcion == "1":
        ejemplo_basico()
    elif opcion == "2":
        ejemplo_webcam()
    else:
        print("Opción no válida")
