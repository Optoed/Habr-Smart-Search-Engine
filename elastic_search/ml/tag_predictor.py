from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
import re

from sklearn.preprocessing import MultiLabelBinarizer


class TagPredictor:
    def __init__(self, model_path='tag_predictor_model.pkl', vectorizer_path='tag_vectorizer.pkl'):
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            self.all_tags = self.model.classes_
            print(f"Модель загружена. Тегов: {len(self.all_tags)}")
        except FileNotFoundError:
            print("Модель не найдена. Запустите обучение.")
            self.model = None
            self.vectorizer = None
        except Exception as e:
            print(f"Ошибка загрузки: {e}")
            self.model = None
            self.vectorizer = None

    def preprocess_text(self, text):
        if not text:
            return ""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def predict_tags(self, title, text, top_n=5):
        if self.model is None or self.vectorizer is None:
            return []

        try:
            combined_text = f"{title} {text}"
            processed_text = self.preprocess_text(combined_text)

            if not processed_text:
                return []

            text_vector = self.vectorizer.transform([processed_text])
            probabilities = self.model.predict_proba(text_vector)[0]

            top_indices = np.argsort(probabilities)[-top_n:][::-1]
            predicted_tags = []

            for idx in top_indices:
                if probabilities[idx] > 0.2:
                    predicted_tags.append({
                        'tag': self.all_tags[idx],
                        'confidence': round(probabilities[idx], 3)
                    })

            return predicted_tags

        except Exception as e:
            print(f"Ошибка предсказания: {e}")
            return []


def extract_tags(tags_str):
    """Извлекает теги из строки"""
    if not isinstance(tags_str, str):
        return []
    clean_tags = tags_str.replace('Теги:', '').replace('теги:', '')
    return [tag.strip() for tag in clean_tags.split(',') if tag.strip()]


def filter_rare_tags(tag_lists, min_count=70):
    """Фильтрует редкие теги, которые встречаются меньше min_count раз"""
    # Считаем частоту тегов
    tag_counter = Counter()
    for tags in tag_lists:
        tag_counter.update(tags)

    # Добавьте в функцию filter_rare_tags:
    print("ТЕГИ, КОТОРЫЕ ЕСТЬ ВО ВСЕХ СТАТЬЯХ:")
    for tag, count in tag_counter.items():
        if count == len(tag_lists):
            print(f"  {tag}: {count}/{len(tag_lists)} статей")

    # И посмотреть на распределение тегов
    print("\nРАСПРЕДЕЛЕНИЕ ТЕГОВ:")
    for tag, count in tag_counter.most_common(60):
        print(f"  {tag}: {count} статей")

    print(f"Всего уникальных тегов до фильтрации: {len(tag_counter)}")

    # Оставляем только частые теги
    frequent_tags = {tag for tag, count in tag_counter.items() if count >= min_count}

    print(f"Тегов после фильтрации (min_count={min_count}): {len(frequent_tags)}")

    # Фильтруем теги в каждом списке
    filtered_tag_lists = []
    for tags in tag_lists:
        filtered_tags = [tag for tag in tags if tag in frequent_tags]
        filtered_tag_lists.append(filtered_tags)

    print("\nТОП 200 САМЫХ ЧАСТЫХ ТЕГОВ:")
    for tag, count in tag_counter.most_common(200):
        print(f"  {tag}: {count} статей")

    return filtered_tag_lists, frequent_tags


def evaluate_model(model, X_test, y_test):
    """Оценка качества модели на тестовой выборке"""

    print("\nОЦЕНКА КАЧЕСТВА МОДЕЛИ:")
    print("=================================")

    # Предсказания вероятностей
    y_pred_proba = model.predict_proba(X_test)

    # Преобразуем в бинарные предсказания по порогу 0.2
    y_pred = (y_pred_proba > 0.2).astype(int)

    print("ОСНОВНЫЕ МЕТРИКИ:")
    print(f"Precision (micro): {precision_score(y_test, y_pred, average='micro', zero_division=0):.4f}")
    print(f"Recall (micro): {recall_score(y_test, y_pred, average='micro', zero_division=0):.4f}")
    print(f"F1-Score (micro): {f1_score(y_test, y_pred, average='micro', zero_division=0):.4f}")


def train_tag_predictor_with_evaluation(df, test_size=0.2, min_tag_count=70):
    print("Обучение модели с разделернием train/test и оценкой...")

    df['combined_text'] = df['title'] + ' ' + df['text']
    df['combined_text'] = df['combined_text'].fillna('').astype(str)

    all_tag_lists = [extract_tags(tags) for tags in df['tags']]
    filtered_tag_lists, frequent_tags = filter_rare_tags(all_tag_lists, min_count=min_tag_count)
    all_tags = sorted(list(frequent_tags))

    print(f"Статей с тегами после фильтрации: {sum(1 for tags in filtered_tag_lists if tags)}/{len(df)}")

    mlb = MultiLabelBinarizer(classes=all_tags)
    y_all = mlb.fit_transform(filtered_tag_lists)

    from logic_regression import time_based_split
    train_df, test_df = time_based_split(df, test_size=test_size)

    y_train = y_all[train_df.index]
    y_test = y_all[test_df.index]

    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.8,
        stop_words=['теги', 'tags', 'hubs', 'blog', 'блог'],
        ngram_range=(1, 2)
    )

    X_train = vectorizer.fit_transform(train_df['combined_text'])
    X_test = vectorizer.transform(test_df['combined_text'])

    model = OneVsRestClassifier(LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'
    ))

    model.fit(X_train, y_train)
    model.classes_ = np.array(all_tags)

    # Оценка качества
    evaluate_model(model, X_test, y_test)

    joblib.dump(model, 'tag_predictor_model.pkl')
    joblib.dump(vectorizer, 'tag_vectorizer.pkl')

    print(f"Модель сохранена")
    return model, vectorizer, all_tags


def main():
    # 1. Загружаем данные для обучения
    try:
        df = pd.read_excel("all_serp_data_for_ml.xlsx")
        print(f"Загружено {len(df)} статей")
    except FileNotFoundError:
        print("Файл all_serp_data_for_ml.xlsx не найден")
        return

    # 2. Проверяем необходимые колонки
    required_columns = ['title', 'text', 'tags']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"Отсутствуют колонки: {missing_columns}")
        return

    print("Запуск обучения модели...")
    train_tag_predictor_with_evaluation(df)

    print("\nОбучение завершено!")


if __name__ == "__main__":
    main()