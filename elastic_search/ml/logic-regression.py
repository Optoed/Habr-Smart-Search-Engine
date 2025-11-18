import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import joblib


def prepare_data(df):
    """Подготовка данных для обучения"""

    # Очистка текстовых полей
    df['title'] = df['title'].fillna('').astype(str)
    df['tags'] = df['tags'].fillna('').astype(str)
    df['hubs'] = df['hubs'].fillna('').astype(str)
    df['query_text'] = df['query_text'].fillna('').astype(str)

    # Создаем комбинированные признаки с учетом запроса
    df['query_title'] = df['query_text'] + ' ' + df['title']
    df['query_tags'] = df['query_text'] + ' ' + df['tags']
    df['query_hubs'] = df['query_text'] + ' ' + df['hubs']

    # Извлекаем числовые признаки
    df['title_length'] = df['title'].str.len()
    df['tags_count'] = df['tags'].str.count(',') + 1

    # Используем score от ElasticSearch как дополнительный признак
    if 'score' in df.columns:
        df['score'] = df['score'].fillna(0)
    else:
        df['score'] = 0

    return df


def train_logistic_regression(df):
    """Обучаем модель логистической регрессии с учетом запроса"""

    print("ОБУЧЕНИЕ ЛОГИСТИЧЕСКОЙ РЕГРЕССИИ")
    print("=" * 50)

    # Подготовка данных
    df = prepare_data(df)

    # Разделяем на признаки и целевую переменную
    # Используем все созданные признаки
    text_features = ['query_title', 'query_tags', 'query_hubs']
    numeric_features = ['title_length', 'tags_count', 'score']

    X = df[text_features + numeric_features]
    y = df['relevance']

    # Разделяем на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Создаем препроцессор для разных типов признаков
    preprocessor = ColumnTransformer(
        transformers=[
            ('query_title_tfidf', TfidfVectorizer(
                max_features=5000,
                stop_words=['теги', 'tags', 'hubs', 'blog', 'блог'],
                min_df=2,
                ngram_range=(1, 2)
            ), 'query_title'),
            ('query_tags_tfidf', TfidfVectorizer(
                max_features=3000,
                stop_words=['теги', 'tags', 'hubs', 'blog', 'блог'],
                min_df=2,
                ngram_range=(1, 1)
            ), 'query_tags'),
            ('query_hubs_tfidf', TfidfVectorizer(
                max_features=2000,
                stop_words=['теги', 'tags', 'hubs', 'blog', 'блог'],
                min_df=2,
                ngram_range=(1, 1)
            ), 'query_hubs'),
            ('numeric', StandardScaler(), numeric_features)
        ]
    )

    # Создаем пайплайн
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            random_state=42,
            max_iter=10000,
            class_weight='balanced'
        ))
    ])

    # Обучаем модель
    print("Обучаем модель с учетом запросов...")
    model.fit(X_train, y_train)

    # Оцениваем качество
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Точность модели: {accuracy:.3f}")
    print("\nОтчет по классификации:")
    print(classification_report(y_test, y_pred))

    return model, X_train, X_test, y_train, y_test, y_pred


def analyze_feature_importance(model, top_n=20):
    """Анализ важности признаков"""

    print(f"\nТОП-{top_n} ВАЖНЫХ ПРИЗНАКОВ:")
    print("=" * 50)

    # Получаем названия признаков после препроцессинга
    feature_names = []
    preprocessor = model.named_steps['preprocessor']

    for name, transformer, columns in preprocessor.transformers_:
        if hasattr(transformer, 'get_feature_names_out'):
            feature_names.extend(transformer.get_feature_names_out())
        else:
            feature_names.extend(columns)

    # Получаем коэффициенты модели
    coefficients = model.named_steps['classifier'].coef_[0]
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': coefficients
    })

    # Сортируем по важности (по модулю)
    feature_importance['abs_importance'] = np.abs(feature_importance['importance'])
    top_features = feature_importance.sort_values('abs_importance', ascending=False).head(top_n)

    for _, row in top_features.iterrows():
        effect = "УВЕЛИЧИВАЕТ" if row['importance'] > 0 else "УМЕНЬШАЕТ"
        print(f"{row['feature']:<30} {effect} вероятность релевантности (вес: {row['importance']:.3f})")

    return top_features

def save_results_to_excel(df, model, X_test, y_test, y_pred):
    """Сохраняет результаты в Excel файл"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ml_model_results_{timestamp}.xlsx"

    # Получаем вероятности для тестовых данных
    test_probs = model.predict_proba(X_test)[:, 1]
    top_features = analyze_feature_importance(model)

    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # 1. Основные метрики
        metrics_data = {
            'Метрика': ['Точность', 'Всего статей', 'Релевантных', 'Нерелевантных'],
            'Значение': [
                f"{accuracy_score(y_test, y_pred):.3f}",
                len(df),
                len(df[df['relevance'] == 1]),
                len(df[df['relevance'] == 0])
            ]
        }
        pd.DataFrame(metrics_data).to_excel(writer, sheet_name='Метрики', index=False)

        # 2. Важные признаки
        top_features.to_excel(writer, sheet_name='Важные признаки', index=False)

        # 3. Тестовые данные с предсказаниями
        test_results = X_test.copy()
        test_results['Фактическая_релевантность'] = y_test.values
        test_results['Предсказанная_релевантность'] = y_pred
        test_results['Вероятность_релевантности'] = test_probs
        test_results['Правильно'] = y_test.values == y_pred

        # Добавляем оригинальные данные
        original_data = df.loc[test_results.index, ['title', 'tags', 'hubs', 'query_text', 'score']]
        test_results = pd.concat([test_results, original_data], axis=1)

        test_results.to_excel(writer, sheet_name='Тестовые данные', index=False)

    print(f"Результаты сохранены в {filename}")
    return filename


def main():
    # Загружаем данные
    df = pd.read_excel("habr_serp_data_llm_with_relevance.xlsx")

    # Проверяем, что есть размеченные данные
    if 'relevance' not in df.columns or df['relevance'].isna().all():
        print("Нет размеченных данных в колонке 'relevance'")
        return

    print(f"Всего статей: {len(df)}")
    print(f"Релевантных: {len(df[df['relevance'] == 1])}")
    print(f"Нерелевантных: {len(df[df['relevance'] == 0])}")

    # Обучаем модель
    model, X_train, X_test, y_train, y_test, y_pred = train_logistic_regression(df)

    # Анализируем важные признаки
    top_features = analyze_feature_importance(model)

    # Демонстрация предсказания на тестовых данных
    print(f"\nДЕМОНСТРАЦИЯ РАБОТЫ МОДЕЛИ:")
    print("=" * 50)

    test_probs = model.predict_proba(X_test)[:, 1]

    # Показываем несколько примеров
    for i in range(min(5, len(X_test))):
        actual = y_test.iloc[i]
        predicted = y_pred[i]
        prob = test_probs[i]

        # Берем оригинальные данные для отображения
        original_idx = X_test.index[i]
        query = df.loc[original_idx, 'query_text'][:30]
        title = df.loc[original_idx, 'title'][:50]

        print(f"\nЗапрос: '{query}...'")
        print(f"Статья: {title}...")
        print(f"Фактическая релевантность: {actual}, Предсказанная: {predicted}")
        print(f"Вероятность релевантности: {prob:.3f}")
        print("Верно" if actual == predicted else "Ошибка")

    # Сохраняем модель для будущего использования
    joblib.dump(model, 'relevance_classifier_with_query.pkl')
    print(f"\nМодель сохранена в 'relevance_classifier_with_query.pkl'")

    # Сохраняем результаты в Excel
    results_file = save_results_to_excel(df, model, X_test, y_test, y_pred)
    print(f"Все результаты сохранены в: {results_file}")


if __name__ == "__main__":
    main()