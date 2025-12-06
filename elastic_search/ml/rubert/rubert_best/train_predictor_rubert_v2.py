import pandas as pd
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    AutoConfig
)
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import re
import numpy as np
import joblib
from collections import Counter
import warnings
import os
from datetime import datetime

# Конфигурация
VERSION = "v2"
MODEL_NAME = "cointegrated/rubert-tiny2"

def time_based_split(df, test_size=0.2):
    # Сортируем по дате
    df_sorted = df.sort_values('date')

    split_idx = int(len(df_sorted) * (1 - test_size))

    # Разделяем вручную
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]

    return train_df, test_df

class MultiLabelBERT(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        # Загружаем базовую модель BERT
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = self.bert.config

        # Классификатор для multi-label
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        # Прямой проход через BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )

        # Берем embedding [CLS] токена
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)

        # Логиты для каждого класса
        logits = self.classifier(pooled_output)

        # Применяем сигмоиду для multi-label
        probabilities = self.sigmoid(logits)

        # Вычисляем loss если есть labels
        loss = None
        if labels is not None:
            loss_fct = nn.BCELoss()  # Binary Cross Entropy для multi-label
            loss = loss_fct(probabilities, labels)

        return {'loss': loss, 'logits': probabilities}


def extract_tags(tags_str):
    """Извлекает теги из строки"""
    if not isinstance(tags_str, str):
        return []

    clean_tags = tags_str.replace('Теги:', '').replace('теги:', '')
    tags = [t.strip() for t in clean_tags.split(',') if t.strip()]

    stop_tags = {'блог', 'blog', 'статья', 'article', 'post', 'запись', 'тег'}
    return [t for t in tags if t.lower() not in stop_tags and len(t) > 1]


def clean_text(t):
    """Очистка текста"""
    if not isinstance(t, str):
        return ""
    t = re.sub(r'[^\w\s.,!?-]', ' ', t)
    t = re.sub(r'\s+', ' ', t)
    return t.strip()


class HabrDataset(torch.utils.data.Dataset):
    """Датасет для обучения"""

    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels.astype(np.float32)  # Важно: float32 для BCELoss
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if not text or len(text.strip()) == 0:
            text = "Пустой текст"

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(self.labels[idx])
        }


def compute_metrics(eval_pred):
    """Метрики для multi-label классификации"""
    predictions, labels = eval_pred
    # predictions уже содержат вероятности после сигмоиды
    predictions_binary = (predictions > 0.2).astype(int)

    precision = precision_score(labels, predictions_binary, average='micro', zero_division=0)
    recall = recall_score(labels, predictions_binary, average='micro', zero_division=0)
    f1 = f1_score(labels, predictions_binary, average='micro', zero_division=0)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    print(f"Начинаем обучение ruBERT для предсказания тегов (версия {VERSION})...")
    print(f"Дата запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Загрузка данных
    try:
        df = pd.read_excel("../../all_serp_data_for_ml.xlsx")
        print(f"Загружено {len(df)} статей")
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        return

    # 2. Подготовка текста

    df["combined_text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).apply(clean_text)

    # Удаляем пустые тексты
    initial_count = len(df)
    df = df[df["combined_text"].str.strip().str.len() > 10].copy()
    print(f"Текст очищен. Осталось {len(df)} статей (удалено {initial_count - len(df)})")

    # 3. Извлечение и фильтрация тегов
    tag_lists = [extract_tags(t) for t in df["tags"]]

    # Статистика тегов
    tag_counter = Counter()
    for tl in tag_lists:
        tag_counter.update(tl)

    print(f"\nСтатистика тегов:")
    print(f"   Всего уникальных тегов: {len(tag_counter)}")
    print(f"   Статей с тегами: {sum(1 for tl in tag_lists if len(tl) > 0)}/{len(df)}")
    print(f"   Теги: {tag_counter.most_common(200)}")

    # Фильтрация редких тегов
    MIN_TAG_COUNT = 90 # 90 подбирал вручную
    freq_tags = {t for t, c in tag_counter.items() if c >= MIN_TAG_COUNT}
    tag_lists = [[t for t in tl if t in freq_tags] for tl in tag_lists]

    # Удаляем статьи без тегов после фильтрации
    valid_indices = [i for i, tl in enumerate(tag_lists) if len(tl) > 0]
    df = df.iloc[valid_indices].reset_index(drop=True)
    tag_lists = [tag_lists[i] for i in valid_indices]

    print(f"\nПосле фильтрации (min_count={MIN_TAG_COUNT}):")
    print(f"   Осталось тегов: {len(freq_tags)}")
    print(f"   Статей с тегами: {len(tag_lists)}")

    print(tag_lists[:200])

    # 4. Преобразование тегов в бинарный формат
    mlb = MultiLabelBinarizer(classes=sorted(list(freq_tags)))
    Y = mlb.fit_transform(tag_lists)

    print(f"\nРазмерность целевой переменной: {Y.shape}")

    # Сохранение кодировщика mlb С СУФФИКСОМ _v2
    mlb_filename = f"rubert_mlb_{VERSION}.pkl"
    joblib.dump(mlb, mlb_filename)
    print(f"MultiLabelBinarizer сохранен в {mlb_filename}")

    # 5. Разделение данных по времени
    # X = df["combined_text"].tolist()
    # X_train, X_test, y_train, y_test = train_test_split(
    #   X, Y, test_size=0.2, random_state=42, shuffle=True # TODO: time split
    # )

    df = df.reset_index(drop=True)

    train_df, test_df = time_based_split(df, test_size=0.2)

    y_train = Y[train_df.index.tolist()]
    y_test = Y[test_df.index.tolist()]

    # Сбрасываем индексы только для текстов
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    X_train = train_df['combined_text']
    X_test = test_df['combined_text']

    print(f"\nРазделение данных:")
    print(f"   Обучающая выборка: {len(X_train)} статей")
    print(f"   Тестовая выборка: {len(X_test)} статей")

    # Сохранение датасетов С СУФФИКСОМ _v2
    joblib.dump((X_train, y_train), f"train_dataset_{VERSION}.pkl")
    joblib.dump((X_test, y_test), f"test_dataset_{VERSION}.pkl")

    # 6. Загрузка токенизатора
    print("\nЗагружаем модель...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print(f"Токенизатор '{MODEL_NAME}' загружен")
    except Exception as e:
        print(f"Ошибка загрузки токенизатора: {e}")
        return

    # 7. Создаем модель
    num_labels = Y.shape[1]
    model = MultiLabelBERT(MODEL_NAME, num_labels)
    print(f"Модель инициализирована ({num_labels} тегов)")

    # 8. Создаем датасеты
    train_dataset = HabrDataset(X_train, y_train, tokenizer, max_len=128)
    test_dataset = HabrDataset(X_test, y_test, tokenizer, max_len=128)

    # 9. Настройка обучения
    output_dir = f"./rubert-tags-results-{VERSION}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=5, # 5 эпох
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir=f'./logs-{VERSION}',
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=False,
        metric_for_best_model="f1",
        greater_is_better=True,
        learning_rate=3e-5,
        fp16=False,
        report_to="none",
        save_total_limit=1,
        dataloader_num_workers=0,
    )

    # 10. Создаем тренер
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("\nНачинаем обучение...")
    print(f"   Устройство: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"   Размер батча: {training_args.per_device_train_batch_size}")
    print(f"   Эпох: {training_args.num_train_epochs}")

    # 11. Обучение
    try:
        trainer.train()
    except Exception as e:
        print(f"Ошибка при обучении: {e}")
        import traceback
        traceback.print_exc()
        return

    # 12. Сохранение модели С СУФФИКСОМ _v2
    try:
        # Сохраняем модель целиком
        model_filename = f"rubert_tag_predictor_{VERSION}.pth"
        torch.save(model.state_dict(), model_filename)

        # Сохраняем конфигурацию
        config_filename = f"rubert_config_{VERSION}.pkl"
        config = {
            'model_name': MODEL_NAME,
            'num_labels': num_labels,
            'max_len': 128,
            'version': VERSION,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'num_tags': len(freq_tags),
            'num_articles': len(df)
        }
        joblib.dump(config, config_filename)

        # Сохраняем токенизатор в папку с суффиксом
        tokenizer_dir = f"./rubert_tokenizer_{VERSION}"
        tokenizer.save_pretrained(tokenizer_dir)

        print("\nОбучение завершено!")
        print("Модель сохранена с суффиксом _v2:")
        print(f"   - {model_filename} - веса модели")
        print(f"   - {config_filename} - конфигурация")
        print(f"   - {tokenizer_dir} - токенизатор")
        print(f"   - {mlb_filename} - кодировщик тегов")
        print(f"   - train_dataset_{VERSION}.pkl - обучающие данные")
        print(f"   - test_dataset_{VERSION}.pkl - тестовые данные")

    except Exception as e:
        print(f"Ошибка при сохранении модели: {e}")
        import traceback
        traceback.print_exc()

    # 13. Финальная оценка
    print("\nФинальная оценка на тестовой выборке:")
    try:
        eval_results = trainer.evaluate()
        for key, value in eval_results.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
    except Exception as e:
        print(f"   Не удалось выполнить оценку: {e}")


# Класс для предсказания
class RubertTagPredictor_v2:
    def __init__(self, model_path=f"rubert_tag_predictor_{VERSION}.pth",
                 config_path=f"rubert_config_{VERSION}.pkl",
                 tokenizer_path=f"./rubert_tokenizer_{VERSION}",
                 mlb_path=f"rubert_mlb_{VERSION}.pkl"):

        print(f"Загружаем модель предсказания тегов (версия {VERSION})...")

        config = joblib.load(config_path)
        print(f"Конфигурация модели:")
        print(f"   Модель: {config['model_name']}")
        print(f"   Тегов: {config['num_labels']}")
        print(f"   Версия: {config['version']}")
        print(f"   Дата создания: {config.get('timestamp', 'N/A')}")

        # Загружаем токенизатор и кодировщик тегов
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.mlb = joblib.load(mlb_path)

        # Создаем и загружаем модель
        self.model = MultiLabelBERT(config['model_name'], config['num_labels'])
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        self.max_len = config.get('max_len', 128)
        self.version = VERSION

        # Перенос на GPU если доступно
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        print(f"Модель загружена на устройство: {self.device}")
        print(f"Доступно тегов: {len(self.mlb.classes_)}")
        print(f"Примеры тегов: {self.mlb.classes_[:10]}...")

    def preprocess_text(self, text):
        if not text:
            return ""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def predict_tags(self, title, text, threshold=0.2, top_n=5):
        """Предсказание тегов для статьи"""
        combined = f"{title} {text}"
        combined = self.preprocess_text(combined)

        print(f"Обрабатываем текст длиной {len(combined)} символов...")

        inputs = self.tokenizer(
            combined,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = outputs['logits'].cpu().numpy()[0]

        print(f"Предсказаны вероятности для {len(probabilities)} тегов")

        top_indices = np.argsort(probabilities)[-top_n:][::-1]
        predicted_tags = []

        for idx in top_indices:
            if probabilities[idx] > threshold:
                tag = self.mlb.classes_[idx]
                predicted_tags.append({
                    'tag': tag,
                    'confidence': float(probabilities[idx])
                })

        print(f"Найдено {len(predicted_tags)} тегов (порог: {threshold})")
        return predicted_tags


def test_predictor():
    """Тестирование обученной модели"""
    print("\nТестирование обученной модели...")

    try:
        predictor = RubertTagPredictor_v2()

        # Тестовые примеры
        test_cases = [
            {
                "title": "Машинное обучение на Python",
                "text": "Искусственный интеллект и нейросети становятся все популярнее. Python является основным языком для ML разработки."
            },
            {
                "title": "Docker контейнеризация",
                "text": "Docker позволяет упаковывать приложения в контейнеры для удобного развертывания. Используется в DevOps практиках."
            },
            {
                "title": "Веб-разработка на JavaScript",
                "text": "JavaScript и фреймворк React используются для создания современных веб-приложений. Frontend разработка."
            }
        ]

        for i, test_case in enumerate(test_cases, 1):
            print(f"\nТест {i}: '{test_case['title']}'")
            tags = predictor.predict_tags(test_case['title'], test_case['text'])
            if tags:
                for tag_info in tags:
                    print(f"   {tag_info['tag']}: {tag_info['confidence']:.3f}")
            else:
                print("   Теги не найдены")

    except Exception as e:
        print(f"Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Запуск обучения + вывод метрик
    main()

    # Тест на выдуманных примерах
    print("\n" + "="*50)
    test_predictor()



