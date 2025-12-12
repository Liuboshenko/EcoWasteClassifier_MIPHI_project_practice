# Проект EcoWasteClassifier

Добро пожаловать в проект EcoWasteClassifier! Это полное, сквозное решение на основе машинного обучения для классификации отходов по категориям переработки. Создано в рамках экологического проекта, оно демонстрирует transfer learning с использованием TensorFlow и Keras для эффективной сортировки изображений мусора.

![Пример AI-системы сортировки отходов в аэропорту SEA](https://www.portseattle.org/sites/default/files/2023-12/231220_Oscar_AI_16x9_WEB.jpg)

## Технический стек

<p align="left">
  <img src="https://img.shields.io/badge/Python_3.10-3776AB?logo=python&logoColor=white" alt="Python 3.10" />
  <img src="https://img.shields.io/badge/Jupyter_Notebook-F37626?logo=jupyter&logoColor=white" alt="Jupyter Notebook" />
  <img src="https://img.shields.io/badge/numpy-013243?logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white" alt="pandas" />
  <img src="https://img.shields.io/badge/matplotlib-11557C?logo=plotly&logoColor=white" alt="matplotlib" />
  <img src="https://img.shields.io/badge/seaborn-4C72B0?logoColor=white" alt="seaborn" />
  <img src="https://img.shields.io/badge/plotly-3F4F75?logo=plotly&logoColor=white" alt="plotly" />
</p>

## Структура проекта
- `README.md`: Этот файл — ваш гид по всему.
- `MIPHI_project_practice_Liuboshenko_v2.ipynb`: Полный ноутбук для обучения.
- `inference.ipynb`: Специальный ноутбук для загрузки и тестирования предобученной модели на новых изображениях.
- `EcoWasteClassifier.h5`: Файл предобученной модели (компактный размер, ~17 МБ).
- `requirements.txt`: Список зависимостей для простой установки.
- `TODO.md`: Планируемые улучшения и будущая работа.
- `examples/`: Папка с несколькими примерами изображений для тестирования (например, cardboard.jpg, glass.jpg — добавьте плейсхолдеры или опишите, как добавить их).

## Решаемая проблема
В современном мире неправильная утилизация отходов значительно способствует загрязнению окружающей среды, переполнению свалок и потере ресурсов. Уровень переработки остается низким частично из-за путаницы в сортировке материалов, таких как картон, стекло, металл, бумага, пластик и обычный мусор. Этот проект решает эту проблему, создавая AI-классификатор изображений, который автоматически определяет типы отходов по фото. Используя реальный датасет (TrashNet), мы обучили модель распознавать эти 6 классов, что позволяет создавать умные приложения для переработки, интеллектуальные контейнеры или образовательные инструменты. Результат? Легковесная, точная модель, работающая на стандартном оборудовании, с точностью валидации около 85-90% (на основе типичных запусков с transfer learning на MobileNetV2). Это способствует устойчивым практикам, делая сортировку отходов доступной и безошибочной.

![AI в управлении отходами: улучшение процессов сортировки и переработки](https://evreka.co/wp-content/uploads/2025/09/waste-srting-e1758206645288.png)

## Что мы достигли
- **Датасет**: TrashNet (2527 изображений по 6 классам: cardboard, glass, metal, paper, plastic, trash).

![Примеры изображений из датасета TrashNet: картон, стекло, металл](https://www.researchgate.net/publication/349188057/figure/fig1/AS:11431281415452702@1746034652591/Images-from-the-TrashNet-dataset-a-cardboard-b-glass-c-metal-d-paper-e.tif)

![Образцы из TrashNet: стекло, бумага, картон](https://www.researchgate.net/publication/349188057/figure/fig1/AS:11431281415452702@1746034652591/Images-from-the-TrashNet-dataset-a-cardboard-b-glass-c-metal-d-paper-e.tif)

- **Архитектура модели**: Transfer learning на базе MobileNetV2 (предобученной на ImageNet), с добавлением кастомных плотных слоев для классификации.

![Диаграмма архитектуры MobileNetV2](https://www.researchgate.net/publication/369624227/figure/fig8/AS:11431281131552328@1680145503560/The-architecture-of-the-MobileNetV2-with-a-sample-input-image-and-19-residual-bottleneck.png)

- **Обучение**: Аугментация данных, ранняя остановка и снижение скорости обучения для надежной производительности. Обучение на GPU для эффективности.

![Графики кривых обучения: точность и потери на тренировке и валидации](https://machinelearningmastery.com/wp-content/uploads/2018/12/Example-of-Train-and-Validation-Learning-Curves-Showing-a-Training-Dataset-the-May-be-too-Small-Relative-to-the-Validation-Dataset.png)

![Линии потерь и точности по эпохам в много-классовой классификации](https://machinelearningmastery.com/wp-content/uploads/2018/11/Line-Plots-of-KL-Divergence-Loss-and-Classification-Accuracy-over-Training-Epochs-on-the-Blobs-Multi-Class-Classification-Problem.png)

- **Выход**: Сохраненная модель Keras (.h5), готовая к развертыванию. Она предсказывает с оценкой уверенности и визуализирует результаты.
- **Удобство использования**: Простые в запуске ноутбуки для обучения и инференса, плюс графики для анализа производительности.

Это не просто демо — это практический инструмент, который можно интегрировать в мобильные приложения или IoT-устройства для сортировки отходов в реальном времени.

![Робот для сортировки пластика на основе AI](https://www.ipi-singapore.org/contents/2024/09/ai-based-material-sorting-robot-for-plastic-recycling-17256040245781/ai-based-material-sorting-robot-for-plastic-recycling.jpeg)

## Инструкции по установке и использованию

### Шаг 1: Клонирование репозитория
Сначала загрузите код на свой локальный компьютер:
```
git clone https://github.com/Liuboshenko/EcoWasteClassifier_MIPHI_project_practice.git
cd EcoWasteClassifier_MIPHI_project_practice.git
```

### Шаг 2: Создание виртуального окружения
Чтобы изолировать зависимости, настройте виртуальное окружение Python3.12:
```
python -m venv env3.12
source env/bin/activate  
```

### Шаг 3: Установка зависимостей
Установите все необходимые пакеты из предоставленного файла:
```
pip install -r requirements.txt
```
Это включает TensorFlow, Keras, Matplotlib, NumPy и другие essentials

### Шаг 4: Обучение модели (опционально)
Если хотите переобучить с нуля:
1. Откройте `MIPHI_project_practice_Liuboshenko_v2.ipynb` в Jupyter Notebook или Google Colab.
2. Запускайте ячейки последовательно (как описано в комментариях ноутбука).
3. Модель скачает датасет, обучит, построит графики и сохранит как `EcoWasteClassifier.h5`.
4. Файл модели автоматически скачается для удобства.

Примечание: В Colab используйте GPU-runtime для ускорения обучения (Runtime > Change runtime type).

### Шаг 5: Использование предобученной модели локально
Для тестирования модели на своих изображениях:
1. Откройте `inference.ipynb` в Jupyter Notebook.
2. Запустите ячейки для загрузки модели.
3. Используйте функцию `predict_waste()`:
   - Вызовите `predict_waste()` для интерактивной загрузки изображения (перетащите в Jupyter).
   - Или передайте путь к файлу: `predict_waste('path/to/your/image.jpg')`.
4. Отобразится изображение с предсказанным классом (на русском в эко-тематике: КАРТОН, СТЕКЛО и т.д.) и оценкой уверенности, с цветовой кодировкой (зеленый >80%, оранжевый 50-80%, красный <50%).

Пример вывода: График с изображением и заголовком вроде "ПЛАСТИК\nУверенность: 92.5%".

### Запуск в Google Colab
- Загрузите файлы репозитория в Colab.
- Установите зависимости с помощью `!pip install -r requirements.txt`.
- Загрузите модель из `EcoWasteClassifier.h5` и тестируйте.

## Пример использования
После настройки, в `inference.ipynb`:
```python
predict_waste('examples/cardboard.jpg')
```
Или интерактивно:
```python
predict_waste()  # Запросит загрузку
```

## Вклад в проект
Проверьте `TODO.md` для идей.

## Лицензия
MIT License — свободно используйте, модифицируйте и распространяйте.
