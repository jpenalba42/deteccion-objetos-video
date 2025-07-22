from flask import Flask, request, render_template, url_for, send_from_directory
import os
import uuid
from detector import difuminar_matricula
from detector_yolo import difuminar_matricula_yolo
from detector_video_yolo import DetectorVideoYOLO

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
OUTPUT_FOLDER = 'procesadas'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

VIDEO_UPLOAD_FOLDER = 'static/uploads/videos'
VIDEO_OUTPUT_FOLDER = 'static/procesadas/videos'
os.makedirs(VIDEO_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEO_OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/imagen', methods=['GET'])
def procesar_imagen():
    return render_template('formulario_imagen.html')

@app.route('/procesar', methods=['POST'])
def procesar():
    if 'imagen' not in request.files:
        return 'No se subió ninguna imagen'

    file = request.files['imagen']
    metodo = request.form.get("metodo", "haar")
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    output_path = os.path.join(OUTPUT_FOLDER, file.filename)
    file.save(input_path)

    if metodo == "yolo":
        hay_placa = difuminar_matricula_yolo(input_path, output_path)
    else:
        hay_placa = difuminar_matricula(input_path, output_path)

    if hay_placa:
        return render_template('resultado.html', imagen_original=file.filename, imagen_procesada=file.filename)
    else:
        return render_template('no_matricula.html')

@app.route('/procesadas/<nombre>')
def mostrar_imagen(nombre):
    return send_from_directory(OUTPUT_FOLDER, nombre)

@app.route('/static/uploads/<nombre>')
def imagen_original(nombre):
    return send_from_directory(UPLOAD_FOLDER, nombre)

@app.route('/video', methods=['GET'])
def procesar_video():
    return render_template('video.html')

@app.route('/procesar_video', methods=['POST'])
def procesar_video_post():
    video = request.files['video']
    difuminar = 'difuminar' in request.form

    if not video:
        return "No se subió ningún video", 400

    # Guardar video subido
    nombre_video = f"{uuid.uuid4()}.mp4"
    path_entrada = os.path.join(VIDEO_UPLOAD_FOLDER, nombre_video)
    video.save(path_entrada)

    # Procesar video
    detector = DetectorVideoYOLO()
    nombre_salida = f"procesado_{nombre_video}"
    path_salida = os.path.join(VIDEO_OUTPUT_FOLDER, nombre_salida)
    detector.detectar_matriculas_video(path_entrada, salida_path=path_salida, difuminar=difuminar, mostrar_video=False)

    return render_template("video_resultado.html",
                           video_url=url_for('static', filename=f'procesadas/videos/{nombre_salida}'),
                           nombre_archivo=nombre_salida)

if __name__ == '__main__':
    app.run(debug=True)