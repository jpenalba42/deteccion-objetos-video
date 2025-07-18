from ultralytics import YOLO
import cv2
import os
import argparse
import numpy as np
from pathlib import Path

class DetectorVideoYOLO:
    def __init__(self, modelo_path=None):
        """
        Inicializa el detector de matrículas para video usando YOLO
        
        Args:
            modelo_path (str): Ruta al modelo YOLO. Si es None, usa el modelo por defecto.
        """
        if modelo_path is None:
            modelo_path = os.path.join("deteccion_matriculas","modelos", "license-plate-finetune-v1l.pt")
        
        if not os.path.exists(modelo_path):
            raise FileNotFoundError(f"No se encontró el modelo en: {modelo_path}")
        
        self.modelo = YOLO(modelo_path)
        print(f"Modelo cargado desde: {modelo_path}")
    
    def detectar_matriculas_video(self, video_path, salida_path=None, mostrar_video=True, 
                                 difuminar=False, confianza=0.5):
        """
        Detecta matrículas en un video
        
        Args:
            video_path (str): Ruta al video de entrada
            salida_path (str): Ruta del video de salida (opcional)
            mostrar_video (bool): Si mostrar el video en tiempo real
            difuminar (bool): Si difuminar las matrículas detectadas
            confianza (float): Umbral de confianza para las detecciones
        """
        # Abrir video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")
        
        # Obtener propiedades del video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Configurar escritor de video si se especifica salida
        out = None
        if salida_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(salida_path, fourcc, fps, (width, height))
        
        frame_count = 0
        detecciones_totales = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Realizar detección
                resultados = self.modelo(frame, conf=confianza, verbose=False)
                
                # Procesar detecciones
                for r in resultados:
                    if r.boxes is not None and len(r.boxes) > 0:
                        for box in r.boxes:
                            # Obtener coordenadas
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            
                            detecciones_totales += 1
                            
                            if difuminar:
                                # Difuminar la región de la matrícula
                                region = frame[y1:y2, x1:x2]
                                if region.size > 0:
                                    region_borrosa = cv2.GaussianBlur(region, (23, 23), 30)
                                    frame[y1:y2, x1:x2] = region_borrosa
                            else:
                                # Dibujar rectángulo y etiqueta
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
                                # Etiqueta con confianza
                                label = f"Matricula: {conf:.2f}"
                                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                                
                                # Fondo para el texto
                                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                                
                                # Texto
                                cv2.putText(frame, label, (x1, y1 - 5), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Mostrar progreso
                if frame_count % 30 == 0:  # Cada 30 frames
                    progreso = (frame_count / total_frames) * 100
                    print(f"Progreso: {progreso:.1f}% - Frame {frame_count}/{total_frames}")
                
                # Guardar frame si hay salida
                if out:
                    out.write(frame)
                
                # Mostrar video
                if mostrar_video:
                    cv2.imshow('Detección de Matrículas', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Detenido por el usuario")
                        break
        
        finally:
            # Limpiar recursos
            cap.release()
            if out:
                out.release()
            if mostrar_video:
                cv2.destroyAllWindows()
        
        print(f"Procesamiento completado:")
        print(f"- Frames procesados: {frame_count}")
        print(f"- Matrículas detectadas: {detecciones_totales}")
        if salida_path:
            print(f"- Video guardado en: {salida_path}")
    
    def detectar_webcam(self, camara_id=0, difuminar=False, confianza=0.5):
        """
        Detecta matrículas en tiempo real desde la webcam
        
        Args:
            camara_id (int): ID de la cámara (0 por defecto)
            difuminar (bool): Si difuminar las matrículas detectadas
            confianza (float): Umbral de confianza para las detecciones
        """
        cap = cv2.VideoCapture(camara_id)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir la cámara: {camara_id}")
        
        print("Detección en tiempo real iniciada. Presiona 'q' para salir.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Realizar detección
                resultados = self.modelo(frame, conf=confianza, verbose=False)
                
                # Procesar detecciones
                matriculas_detectadas = 0
                for r in resultados:
                    if r.boxes is not None and len(r.boxes) > 0:
                        for box in r.boxes:
                            matriculas_detectadas += 1
                            
                            # Obtener coordenadas
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            
                            if difuminar:
                                # Difuminar la región de la matrícula
                                region = frame[y1:y2, x1:x2]
                                if region.size > 0:
                                    region_borrosa = cv2.GaussianBlur(region, (23, 23), 30)
                                    frame[y1:y2, x1:x2] = region_borrosa
                            else:
                                # Dibujar rectángulo y etiqueta
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
                                # Etiqueta con confianza
                                label = f"Matricula: {conf:.2f}"
                                cv2.putText(frame, label, (x1, y1 - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Mostrar información en pantalla
                info_text = f"Matriculas detectadas: {matriculas_detectadas}"
                cv2.putText(frame, info_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Mostrar frame
                cv2.imshow('Detección de Matrículas - Webcam', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Detector de matrículas en video usando YOLO')
    parser.add_argument('--video', type=str, help='Ruta al video de entrada')
    parser.add_argument('--webcam', action='store_true', help='Usar webcam en tiempo real')
    parser.add_argument('--camara', type=int, default=0, help='ID de la cámara (default: 0)')
    parser.add_argument('--salida', type=str, help='Ruta del video de salida')
    parser.add_argument('--modelo', type=str, help='Ruta al modelo YOLO personalizado')
    parser.add_argument('--difuminar', action='store_true', help='Difuminar matrículas detectadas')
    parser.add_argument('--confianza', type=float, default=0.5, help='Umbral de confianza (default: 0.5)')
    parser.add_argument('--no-mostrar', action='store_true', help='No mostrar video durante procesamiento')
    
    args = parser.parse_args()
    
    # Validar argumentos
    if not args.webcam and not args.video:
        print("Error: Debe especificar --video o --webcam")
        return
    
    try:
        # Crear detector
        detector = DetectorVideoYOLO(args.modelo)
        
        if args.webcam:
            # Detección en webcam
            detector.detectar_webcam(
                camara_id=args.camara,
                difuminar=args.difuminar,
                confianza=args.confianza
            )
        else:
            # Detección en video
            if not os.path.exists(args.video):
                print(f"Error: No se encontró el video: {args.video}")
                return
            
            detector.detectar_matriculas_video(
                video_path=args.video,
                salida_path=args.salida,
                mostrar_video=not args.no_mostrar,
                difuminar=args.difuminar,
                confianza=args.confianza
            )
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
