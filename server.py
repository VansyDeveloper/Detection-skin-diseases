import os
from datetime import datetime
from flask import Flask, render_template, request, send_from_directory, abort, redirect
from PIL import Image

import torch
import torchvision
import torchvision.transforms.functional

from src.model import Model

import cv2

# Выбираем лучшее устройство для выполнения вычислений.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Названия категории.
CATEGORIES = [
    "Угревая сыпь или розацея.",
    "Актинический кератоз, базалиома и другие злокачественные новообразования.",
    "Атопический дерматит.",
]

# Количество классов которые предсказывает модель.
NUM_CLASSES = 3

# Размер образца.
SAMPLE_SIZE = 128

# Функция активации.
ACTIVATION = "silu"

# Карта слоёв и блоков (Residual Blocks) модели.
# Первый уровень определяет количество ступеней свёртки,
# после каждой из которых за исключением последней разрешение
# карт признаков будет уменьшено в два раза (downsampling)
# Каждая ступень задаёт количество блоков с остаточной связью (ResidualBlock)
# где значение описывает количество карт признаков (out_channels) на выходе блока.
BLOCKS = [
    [64, 64, 64, 128], # 32 -> 16
    [128, 128, 128, 128, 256], # 16 -> 8
    [256, 256, 256, 256, 256, 256], # 8 -> 4 
]

# Вероятность dropout в блоках ResidualBlock.
DROPOUT = 0.2

# Тип нормализации "group" или "batch"
NORMALIZATION: str = "group"

# Количество групп.
NUM_GROUPS: int = 16

# Momentum.
MOMENTUM: float = 0.1

# Значение, добавляемое к знаменателю для численной устойчивости.
EPS: float = 1e-6

# Обучаемые параметры в нормализации.
AFFINE: bool = True

# Модель.
model = Model(
    num_classes=NUM_CLASSES,
    blocks = BLOCKS,    
    activation=ACTIVATION,
    dropout=DROPOUT,
    normalization=NORMALIZATION,
    num_groups=NUM_GROUPS,
    momentum=MOMENTUM,
    eps=EPS,
    affine=AFFINE,
).to(DEVICE)

checkpoint = torch.load("./model.pt", map_location=DEVICE, weights_only = True)
model.load_state_dict(checkpoint['model.state_dict'])

model.eval()

# Папка для сохранения образцов.
UPLOAD_FOLDER = "static/uploads"

# Размер стороны образца (квадрат).
SAMPLE_SIZE = 128

app = Flask(
    __name__,
    static_folder="static",
)

# Пути к моделям для анализа пола и возраста
FACE_WEIGHTS = "opencv_face_detector.pbtxt"
FACE_MODEL = "opencv_face_detector_uint8.pb"
GENDER_PROTO = "gender_deploy.prototxt"
GENDER_MODEL = "gender_net.caffemodel"
AGE_PROTO = "age_deploy.prototxt"
AGE_MODEL = "age_net.caffemodel"

# Загрузка моделей
face_network = cv2.dnn.readNet(FACE_MODEL, FACE_WEIGHTS)
gender_network = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
age_network = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)

# Константы для анализа пола и возраста
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
GENDER_LIST = ['Male', 'Female']
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

def analyze_gender_and_age(image):
    # Преобразование изображения в формат OpenCV
    image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Поиск лица
    height, width = image_cv.shape[:2]
    blob = cv2.dnn.blobFromImage(image_cv, 1.0, (300, 300), [104, 117, 123], True, False)
    face_network.setInput(blob)
    detections = face_network.forward()

    # Инициализация переменных
    gender, age = "Unknown", "Unknown"

    # Обработка результатов детекции лица
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.8:  # Порог уверенности
            x1 = int(detections[0, 0, i, 3] * width)
            y1 = int(detections[0, 0, i, 4] * height)
            x2 = int(detections[0, 0, i, 5] * width)
            y2 = int(detections[0, 0, i, 6] * height)

            # Вырезаем лицо
            face = image_cv[max(0, y1):min(y2, height), max(0, x1):min(x2, width)]
            if face.size == 0:
                continue

            # Анализ пола
            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            gender_network.setInput(face_blob)
            gender_pred = gender_network.forward()
            gender = GENDER_LIST[gender_pred[0].argmax()]

            # Анализ возраста
            age_network.setInput(face_blob)
            age_pred = age_network.forward()
            age = AGE_LIST[age_pred[0].argmax()][1:-1]  # Убираем скобки

            # Прерываем цикл после первого найденного лица
            break

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
        # Преобразуем загруженный файл в PILImage
        im = Image.open(f.stream)

        # Получаем образец изображения требуемого размера
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
        filename = os.path.join(folder, f"{index}.jpg")
        im.save(filename, format='JPEG', subsampling=0, quality=100)

        # Конвертируем картинку в тензор
        t = torchvision.transforms.functional.pil_to_tensor(im) / 255
        t = t.to(DEVICE)
        t = t.unsqueeze(dim=0)

        # Предсказание заболевания
        with torch.no_grad():
            prediction = model(t)
        
        p_index = torch.argmax(prediction).item()
        predicted_value = prediction[0, p_index]

        # Анализ пола и возраста
        image_cv = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        gender, age = analyze_gender_and_age(image_cv)

        # Формируем тело отчёта
        report += "<div class=\"card\">"
        report += "<div class=\"image_area\">"
        report += f"<img src=\"{filename}\" />"
        report += "</div>"
        
        report += "<div class=\"content\">"
        report += f"<div class=\"prediction\"><span class=\"light-gray-text\">Диагноз:</span> {CATEGORIES[p_index]}</div>"
        report += f"<div class=\"predicted_value\"><span class=\"light-gray-text\">Предсказанное значение:</span> {predicted_value:.4f}</div>"
        report += f"<div class=\"gender\"><span class=\"light-gray-text\">Пол:</span> {gender}</div>"
        report += f"<div class=\"age\"><span class=\"light-gray-text\">Возраст:</span> {age} лет</div>"
        report += "</div>"

        report += "</div>"

    return render_template("predict.html", content=report)
