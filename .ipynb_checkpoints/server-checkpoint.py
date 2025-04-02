import os
from datetime import datetime
from flask import Flask, render_template, request, redirect
from PIL import Image
import torch
import torchvision
import torchvision.transforms.functional
from src.model import Model
import cv2
from werkzeug.utils import secure_filename
import logging
from threading import Thread
import gc

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)

# Константы
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CATEGORIES = [
    "Угревая сыпь или розацея.",
    "Актинический кератоз, базалиома и другие злокачественные новообразования.",
    "Атопический дерматит.",
]
NUM_CLASSES = 3
SAMPLE_SIZE = 128
UPLOAD_FOLDER = "static/uploads"

# Создание экземпляра Flask-приложения
app = Flask(__name__, static_folder="static")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Ограничение размера загружаемых файлов (16 МБ)

# Модель
model = Model(
    num_classes=NUM_CLASSES,
    blocks=[[64, 64, 64, 128], [128, 128, 128, 128, 256], [256, 256, 256, 256, 256, 256]],
    activation="silu",
    dropout=0.2,
    normalization="group",
    num_groups=16,
    momentum=0.1,
    eps=1e-6,
    affine=True,
).to(DEVICE)

checkpoint = torch.load("./model.pt", map_location=DEVICE, weights_only=True)
model.load_state_dict(checkpoint['model.state_dict'])
model.eval()

# Модели OpenCV
FACE_WEIGHTS = "opencv_face_detector.pbtxt"
FACE_MODEL = "opencv_face_detector_uint8.pb"
GENDER_PROTO = "gender_deploy.prototxt"
GENDER_MODEL = "gender_net.caffemodel"
AGE_PROTO = "age_deploy.prototxt"
AGE_MODEL = "age_net.caffemodel"

face_network = cv2.dnn.readNet(FACE_MODEL, FACE_WEIGHTS)
gender_network = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
age_network = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)

if face_network.empty() or gender_network.empty() or age_network.empty():
    logging.error("Ошибка: Не удалось загрузить одну из моделей OpenCV.")
    raise ValueError("Не удалось загрузить модели OpenCV.")

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
GENDER_LIST = ['Male', 'Female']
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

def analyze_gender_and_age(image):
    image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    height, width = image_cv.shape[:2]
    blob = cv2.dnn.blobFromImage(image_cv, 1.0, (300, 300), [104, 117, 123], True, False)
    face_network.setInput(blob)
    detections = face_network.forward()

    gender, age = "Unknown", "Unknown"
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.8:
            x1 = int(detections[0, 0, i, 3] * width)
            y1 = int(detections[0, 0, i, 4] * height)
            x2 = int(detections[0, 0, i, 5] * width)
            y2 = int(detections[0, 0, i, 6] * height)
            face = image_cv[max(0, y1):min(y2, height), max(0, x1):min(x2, width)]
            if face.size == 0:
                continue

            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            gender_network.setInput(face_blob)
            gender_pred = gender_network.forward()
            gender = GENDER_LIST[gender_pred[0].argmax()]

            age_network.setInput(face_blob)
            age_pred = age_network.forward()
            age = AGE_LIST[age_pred[0].argmax()][1:-1]
            break

    del image_cv
    del face_blob
    gc.collect()
    return gender, age

@app.route("/")
def main():
    return render_template("index.html")

@app.post("/predict")
def predict():
    report = ""
    files = request.files.getlist("files")
    if len(files) == 1 and files[0].filename == "":
        return redirect("/")

    for index, f in enumerate(files):
        if not f.content_type.startswith("image/"):
            report += "<div class=\"error\">Ошибка: Загруженный файл не является изображением.</div>"
            continue

        try:
            # Сохраняем файл
            filename = secure_filename(f.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(file_path)

            # Открываем изображение
            with Image.open(file_path) as im:
                logging.debug(f"Successfully opened image: {f.filename}")

                a = SAMPLE_SIZE / im.width if im.width < im.height else SAMPLE_SIZE / im.height
                w = int(im.width * a)
                h = int(im.height * a)
                im = im.resize((w, h), Image.Resampling.LANCZOS)
                x = (im.width - im.height) // 2 if im.width > im.height else 0
                y = (im.height - im.width) // 2 if im.height > im.width else 0
                im = im.crop((x, y, x + SAMPLE_SIZE, y + SAMPLE_SIZE))
                im = im.convert("RGB")

                # Сохраняем образцы на диск
                sess_folder = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                folder = os.path.join(UPLOAD_FOLDER, sess_folder)
                os.makedirs(folder, exist_ok=True)
                saved_filename = os.path.join(folder, f"{index}.jpg")
                im.save(saved_filename, format='JPEG', subsampling=0, quality=100)
                logging.debug(f"Image saved to: {saved_filename}")

                # Конвертируем картинку в тензор
                t = torchvision.transforms.functional.pil_to_tensor(im) / 255
                t = t.to(DEVICE)
                t = t.unsqueeze(dim=0)

                # Предсказание заболевания
                #with torch.no_grad():
                prediction = model(t)
                p_index = torch.argmax(prediction).item()
                predicted_value = prediction[0, p_index]

                # Анализ пола и возраста
                image_cv = cv2.cvtColor(cv2.imread(saved_filename), cv2.COLOR_BGR2RGB)
                gender, age = analyze_gender_and_age(image_cv)

                # Формируем тело отчёта
                report += "<div class=\"card\">"
                report += "<div class=\"image_area\">"
                report += f"<img src=\"{saved_filename}\" />"
                report += "</div>"
                report += "<div class=\"content\">"
                report += f"<div class=\"prediction\"><span class=\"light-gray-text\">Диагноз:</span> {CATEGORIES[p_index]}</div>"
                report += f"<div class=\"predicted_value\"><span class=\"light-gray-text\">Предсказанное значение:</span> {predicted_value:.4f}</div>"
                report += f"<div class=\"gender\"><span class=\"light-gray-text\">Пол:</span> {gender}</div>"
                report += f"<div class=\"age\"><span class=\"light-gray-text\">Возраст:</span> {age} лет</div>"
                report += "</div>"
                report += "</div>"

        except Exception as e:
            logging.error(f"Error processing file {f.filename}: {str(e)}")
            report += f"<div class=\"error\">Ошибка при обработке изображения: {str(e)}</div>"
            continue

    return render_template("predict.html", content=report)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)