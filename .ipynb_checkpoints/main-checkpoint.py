import os
from datetime import datetime
from flask import Flask, render_template, request, send_from_directory, abort, redirect
from PIL import Image

import torch
import torchvision
import torchvision.transforms.functional

from src.model import Model

# Выбираем лучшее устройство для выполнения вычислений.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Названия категории.
CLASSES = [
    "Угревая сыпь или розацея.",
    "Актинический кератоз, базалиома и другие злокачественные новообразования.",
    "Атопический дерматит.",
    "Диагноз не установлен.",
]

# Количество классов которые предсказывает модель.
NUM_CLASSES = 4

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

checkpoint = torch.load("./model.pt", weights_only = True)
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

        # Получаем образец изображения требуемого нам размера SAMPLE_SIZE x SAMPLE_SIZE.
        # Устанавливаем размер изображения равным SAMPLE_SIZE по меньшей стороне.
        # Вырезаем образец по центру.
        a = SAMPLE_SIZE / im.width if im.width < im.height else SAMPLE_SIZE / im.height
        w = int(im.width * a)
        h = int(im.height * a)
        im = im.resize((w, h), Image.Resampling.LANCZOS)
        x = (im.width - im.height) // 2 if im.width > im.height else 0
        y = (im.height - im.width) // 2 if im.height > im.width else 0
        im = im.crop((x, y, x + SAMPLE_SIZE, y + SAMPLE_SIZE))
        im = im.convert("RGB")

        # Сохраняем образцы на диск.
        sess_folder = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = os.path.join(UPLOAD_FOLDER, sess_folder)
        os.makedirs(folder, exist_ok=True) 
        filename = os.path.join(folder, f"{index}.jpg")
        im.save(filename, format='JPEG', subsampling=0, quality=100)

        # Конвертируем картинку в тензор с значениями от 0 до 1.
        t = torchvision.transforms.functional.pil_to_tensor(im) / 255
        t = t.to(DEVICE)
        t = t.unsqueeze(dim=0)

        prediction = model(t)
        p_index = prediction.argmax().item()
        predicted_value = prediction[0, p_index]

        # Определяем имя диагноза.
        diagnosis = CLASSES[p_index]

        diagnosis = f"{diagnosis} - {p_index}"

        if predicted_value < 0.6:
            diagnosis = "Диагноз не установлен."

        # Формируем тело отчёта.
        report += "<div class=\"card\">"

        report += "<div class=\"image_area\">"
        report += f"<img src=\"{filename}\" />"
        report += "</div>"
        
        report += "<div class=\"content\">"
        report += f"<div class=\"prediction\"><span class=\"light-gray-text\">Диагноз:</span> {diagnosis}</div>"
        report += f"<div class=\"predicted_value\"><span class=\"light-gray-text\">Предсказанное значение:</span> {predicted_value:.4f}</div>"
        report += "</div>"

        report += "</div>"

    return render_template("predict.html", content=report)
