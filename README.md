# Глубокое обучение: сегментация зубов по снимкам компьютерной томографии (КТ)
## Задача предсказать 42 класса зубов (лейблы указаны в json файле набора данных)

![overview](https://github.com/user-attachments/assets/74693649-c159-4d40-9517-1116c4a49300)


## Инструкции по запуску:

- Конфигурация нейронной сети nnUNetv2 адаптирована под различные форматы данных (3D, KT) (https://github.com/MIC-DKFZ/nnUNet)
- Набор данных, тут же лежат классы для предсказания (https://ditto.ing.unimore.it/toothfairy2/)
- Предобработка набора данных
- Обучение модели
- Результаты и оптимизация
- Контейнеризация и упаковка в Docker

<img width="1201" alt="predict" src="https://github.com/user-attachments/assets/f4001f0c-b937-4196-888e-210e50f9d738">

## Как запустить:
### 1. nnUNet Конфигурация
Репозиторий nnUnet: https://github.com/MIC-DKFZ/nnUNet 
```
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```
### 2. Набор данных

Загрузка ToothFairy2 Dataset https://ditto.ing.unimore.it/toothfairy2/

Разархивировать набор данных и положить в *nnUNet_raw* папку.

Заменить *dataset.json* на *preprocess/dataset.json* .

### 3. Предобработка

Подготовка набора данных.

```
python preprocess/preprocess.py --input_root YOUR_nnUNet_FOLDER/nnUNet_raw/Dataset112_ToothFairy2
```

Отфильтруйте неоптимальные изображения и аннотации (необязательно, если это так, измените numTraining в *dataset.json*).

Проведите автоматическую предварительную обработку с помощью nnUNet.

```
nnUNetv2_plan_and_preprocess -d 112 --verify_dataset_integrity
```

### 4. Обучение

Обучение с помощью nnUNetv2 с использованием метода 5-кратной перекрестной проверки.
Run script:

```
sh train.sh
```

### 5. Adaptive Structure Optimization

Чтобы получить оптимальный размер фильтрации, запустите:

```
python utils/ASO.py --gt_root GROUND_TRUTH_FOLDER --pred_root EVALUATION_RESULT_FOLDER
```

### 6. Контейнеризация модели в Docker

Переместите checkpoint_best.pth для каждой папки в *weight*.

Или вы можете получить веса обученной модели из https://drive.google.com/file/d/1OXf4JAugFRThhNCdnnALLivzhRi34eGE/view?usp=sharing

Создайте docker_compose на основе `dockerfile`

```
sh build.sh
```
Save the docker images if needed.
```
sh export.sh
```

## Prediction labels 42:

```
"labels": {
    "background": 0,
    "Lower Jawbone": 1,
    "Upper Jawbone": 2,
    "Left Inferior Alveolar Canal": 3,
    "Right Inferior Alveolar Canal": 4,
    "Left Maxillary Sinus": 5,
    "Right Maxillary Sinus": 6,
    "Pharynx": 7,
    "Bridge": 8,
    "Crown": 9,
    "Implant": 10,
    "Upper Right Central Incisor": 11,
    "Upper Right Lateral Incisor": 12,
    "Upper Right Canine": 13,
    "Upper Right First Premolar": 14,
    "Upper Right Second Premolar": 15,
    "Upper Right First Molar": 16,
    "Upper Right Second Molar": 17,
    "Upper Right Third Molar (Wisdom Tooth)": 18,
    "Upper Left Central Incisor": 19,
    "Upper Left Lateral Incisor": 20,
    "Upper Left Canine": 21,
    "Upper Left First Premolar": 22,
    "Upper Left Second Premolar": 23,
    "Upper Left First Molar": 24,
    "Upper Left Second Molar": 25,
    "Upper Left Third Molar (Wisdom Tooth)": 26,
    "Lower Left Central Incisor": 27,
    "Lower Left Lateral Incisor": 28,
    "Lower Left Canine": 29,
    "Lower Left First Premolar": 30,
    "Lower Left Second Premolar": 31,
    "Lower Left First Molar": 32,
    "Lower Left Second Molar": 33,
    "Lower Left Third Molar (Wisdom Tooth)": 34,
    "Lower Right Central Incisor": 35,
    "Lower Right Lateral Incisor": 36,
    "Lower Right Canine": 37,
    "Lower Right First Premolar": 38,
    "Lower Right Second Premolar": 39,
    "Lower Right First Molar": 40,
    "Lower Right Second Molar": 41,
    "Lower Right Third Molar (Wisdom Tooth)": 42
  }
```
