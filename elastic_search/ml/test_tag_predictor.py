import pandas as pd
from tag_predictor import TagPredictor


def test_with_sample_articles():
    """Тестируем модель на заранее подготовленных примерах"""
    print("🧪 ТЕСТИРОВАНИЕ МОДЕЛИ ПРЕДСКАЗАНИЯ ТЕГОВ")
    print("=" * 60)

    # Загружаем модель
    try:
        predictor = TagPredictor()
        print("✅ Модель успешно загружена")
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return

    # Тестовые статьи
    test_articles = [
        {
            'title': 'Машинное обучение на Python с scikit-learn',
            'text': '''
            В этой статье рассмотрим основы машинного обучения с использованием библиотеки scikit-learn. 
            Поговорим о линейной регрессии, классификации, кластеризации и нейронных сетях. 
            Примеры кода на Python помогут лучше понять материал.
            '''
        },
        {
            'title': 'Разработка веб-приложений на Django',
            'text': '''
            Django - мощный фреймворк для создания веб-приложений на Python. 
            Рассмотрим модели, представления, шаблоны, ORM и работу с базами данных. 
            Создадим простое CRUD-приложение с аутентификацией пользователей.
            '''
        },
        {
            'title': 'Оптимизация запросов в PostgreSQL',
            'text': '''
            Как ускорить работу базы данных PostgreSQL с помощью индексов, 
            анализа запросов EXPLAIN и оптимизации SQL. Рассмотрим работу с большими объемами данных 
            и методы повышения производительности.
            '''
        },
        {
            'title': 'React и Redux для фронтенд разработки',
            'text': '''
            Современная фронтенд разработка с использованием React, Redux и TypeScript. 
            Компонентный подход, управление состоянием приложения, работа с API. 
            Создадим одностраничное приложение с маршрутизацией.
            '''
        },
        {
            'title': 'Docker и контейнеризация приложений',
            'text': '''
            Docker позволяет упаковывать приложения в контейнеры для простого развертывания. 
            Изучим Dockerfile, docker-compose, volumes и сети. 
            Настроим микросервисную архитектуру с помощью контейнеров.
            '''
        }
    ]

    print("\n📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    print("=" * 60)

    for i, article in enumerate(test_articles, 1):
        print(f"\n{i}. 📝 Статья: {article['title']}")
        print("-" * 50)

        # Предсказываем теги
        predicted_tags = predictor.predict_tags(article['title'], article['text'])

        if predicted_tags:
            print("🏷️  Предсказанные теги:")
            for tag_info in predicted_tags:
                confidence_bar = "█" * int(tag_info['confidence'] * 20)
                print(f"   - {tag_info['tag']:25} {confidence_bar} ({tag_info['confidence']:.3f})")
        else:
            print("❌ Теги не предсказаны")


def test_with_real_data():
    """Тестируем на реальных данных из файла"""
    print("\n\n🎯 ТЕСТИРОВАНИЕ НА РЕАЛЬНЫХ ДАННЫХ")
    print("=" * 60)

    try:
        # Загружаем реальные данные
        df = pd.read_excel("all_serp_data_for_ml.xlsx")
        print(f"✅ Загружено {len(df)} статей для тестирования")

        # Загружаем модель
        predictor = TagPredictor()

        # Берем несколько случайных статей с тегами
        articles_with_tags = df[df['tags'].notna()].head(3)

        if len(articles_with_tags) == 0:
            print("❌ Нет статей с тегами для тестирования")
            return

        for idx, row in articles_with_tags.iterrows():
            print(f"\n📝 Реальная статья: {row['title'][:80]}...")
            print("   Фактические теги:", row['tags'])

            # Предсказываем теги
            text = row.get('text', '')[:1000]  # Берем первые 1000 символов
            predicted_tags = predictor.predict_tags(row['title'], text)

            if predicted_tags:
                print("   🏷️  Предсказанные теги:")
                for tag_info in predicted_tags[:5]:  # Показываем топ-5
                    print(f"      - {tag_info['tag']} ({tag_info['confidence']:.3f})")
            else:
                print("   ❌ Теги не предсказаны")
            print("-" * 50)

    except Exception as e:
        print(f"❌ Ошибка при тестировании на реальных данных: {e}")


def test_performance():
    """Тестируем производительность модели"""
    print("\n\n⚡ ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 60)

    import time

    predictor = TagPredictor()

    # Тестовая статья
    test_article = {
        'title': 'Тест производительности модели',
        'text': 'Python машинное обучение данные анализ нейросети глубокое обучение'
    }

    # Замер времени предсказания
    start_time = time.time()

    for i in range(10):
        predicted_tags = predictor.predict_tags(test_article['title'], test_article['text'])

    end_time = time.time()
    avg_time = (end_time - start_time) / 10

    print(f"Среднее время предсказания: {avg_time:.4f} секунд")
    print(f"Тегов предсказано: {len(predicted_tags)}")
    print("Пример предсказания:", [tag['tag'] for tag in predicted_tags])


def main():
    """Основная функция тестирования"""
    print("🎯 ТЕСТИРОВАНИЕ МОДЕЛИ ПРЕДСКАЗАНИЯ ТЕГОВ")
    print("=" * 60)

    # 1. Тест на подготовленных примерах
    test_with_sample_articles()

    # 2. Тест на реальных данных
    test_with_real_data()

    # 3. Тест производительности
    test_performance()

    print("\n" + "=" * 60)
    print("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")


if __name__ == "__main__":
    main()