import joblib
import pandas as pd


class MLRelevanceRanker:
    def __init__(self, model_path='./ml/relevance_classifier_with_query.pkl'):
        """Инициализация ML-ранкера"""
        self.model = joblib.load(model_path)

    def prepare_article_data(self, query_text, article_data):
        """Подготавливает данные статьи для ML-модели"""
        data = {
            'title': article_data.get('title', ''),
            'tags': article_data.get('tags', ''),
            'hubs': article_data.get('hubs', ''),
            'query_text': query_text,
            'score': article_data.get('_score', 0)
        }
        return pd.DataFrame([data])

    def calculate_ml_score(self, query_text, article_data):
        """Вычисляет ML-скор релевантности"""
        try:
            # Подготавливаем данные
            df = self.prepare_article_data(query_text, article_data)

            # Создаем комбинированные признаки (как при обучении)
            df['query_title'] = df['query_text'] + ' ' + df['title']
            df['query_tags'] = df['query_text'] + ' ' + df['tags']
            df['query_hubs'] = df['query_text'] + ' ' + df['hubs']
            df['title_length'] = df['title'].str.len()
            df['tags_count'] = df['tags'].str.count(',') + 1

            # Признаки для предсказания
            text_features = ['query_title', 'query_tags', 'query_hubs']
            numeric_features = ['title_length', 'tags_count', 'score']
            X = df[text_features + numeric_features]

            # Предсказываем вероятность релевантности
            ml_score = self.model.predict_proba(X)[0, 1]
            return float(ml_score)

        except Exception as e:
            print(f"Ошибка ML-ранжирования: {e}")
            return 0.0

    def rerank_results(self, query_text, es_results, ml_weight=0.7, es_weight=0.3):
        """
        Переранжирует результаты ElasticSearch с помощью ML-модели

        Args:
            query_text: поисковый запрос
            es_results: результаты от ElasticSearch
            ml_weight: вес ML-скора (0-1)
            es_weight: вес ES-скора (0-1)
        """
        if not es_results or 'hits' not in es_results:
            return es_results

        hits = es_results['hits']['hits']

        # Добавляем ML-скор к каждому результату
        enhanced_results = []
        for hit in hits:
            article_data = {
                'title': hit['_source'].get('title', ''),
                'tags': ', '.join(hit['_source'].get('tags', [])),
                'hubs': ', '.join(hit['_source'].get('hubs', [])),
                '_score': hit['_score']
            }

            # Вычисляем ML-скор
            ml_score = self.calculate_ml_score(query_text, article_data)

            # Нормализуем ES score (приводим к 0-1)
            es_score_normalized = min(hit['_score'] / 100, 1.0)

            # Комбинированный score
            combined_score = (ml_weight * ml_score +
                              es_weight * es_score_normalized)

            enhanced_results.append({
                **hit, # распаковка словаря
                '_ml_score': ml_score,
                '_combined_score': combined_score
            })

        # Сортируем по комбинированному score
        enhanced_results.sort(key=lambda x: x['_combined_score'], reverse=True)

        # Обновляем исходную структуру результатов
        es_results['hits']['hits'] = enhanced_results

        return es_results