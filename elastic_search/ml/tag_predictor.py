import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
import re


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
                if probabilities[idx] > 0.1:
                    predicted_tags.append({
                        'tag': self.all_tags[idx],
                        'confidence': round(probabilities[idx], 3)
                    })

            return predicted_tags

        except Exception as e:
            print(f"Ошибка предсказания: {e}")
            return []

def train_tag_predictor(df, output_model='tag_predictor_model.pkl', output_vectorizer='tag_vectorizer.pkl'):
    print("Обучение модели...")

    df['combined_text'] = df['title'] + ' ' + df['text']
    df['combined_text'] = df['combined_text'].fillna('').astype(str)

    all_tags = set()
    for tags_str in df['tags'].dropna():
        if isinstance(tags_str, str):
            clean_tags = tags_str.replace('Теги:', '').replace('теги:', '')
            tags_list = [tag.strip() for tag in clean_tags.split(',') if tag.strip()]
            all_tags.update(tags_list)

    all_tags = sorted(list(all_tags))
    print(f"Уникальных тегов: {len(all_tags)}")

    y = []
    for _, row in df.iterrows():
        tags_present = [0] * len(all_tags)
        if isinstance(row['tags'], str):
            clean_article_tags = row['tags'].replace('Теги:', '').replace('теги:', '')
            article_tags = [tag.strip() for tag in clean_article_tags.split(',') if tag.strip()]
            for i, tag in enumerate(all_tags):
                if tag in article_tags:
                    tags_present[i] = 1
        y.append(tags_present)

    y = np.array(y)

    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.8,
        stop_words=['теги', 'tags', 'hubs', 'blog', 'блог'],
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(df['combined_text'])

    model = OneVsRestClassifier(LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'
    ))

    model.fit(X, y)
    model.classes_ = np.array(all_tags)

    joblib.dump(model, output_model)
    joblib.dump(vectorizer, output_vectorizer)

    print(f"Модель сохранена: {output_model}")
    return model, vectorizer, all_tags