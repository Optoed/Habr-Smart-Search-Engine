import json
import pandas as pd

from elastic_search.habr_search import HabrSearchEngine

# Конфигурация тестовых запросов
test_queries_config = [
    # Simple поиск - без бустинга, присутствует хотя бы 67% слов
    {'query': 'python машинное обучение', 'type': 'simple'},
    {'query': 'docker контейнеризация', 'type': 'simple'},
    {'query': 'нейросети искусственный интеллект', 'type': 'simple'},
    {'query': 'веб разработка frontend', 'type': 'simple'},
    {'query': 'база данных SQL', 'type': 'simple'},
    {'query': 'пайтон программирование', 'type': 'simple'},
    {'query': 'javascript фреймворки', 'type': 'simple'},
    {'query': 'golang новичку', 'type': 'simple'},
    {'query': 'мобильные приложения', 'type': 'simple'},
    {'query': 'разработка игр unity', 'type': 'simple'},
]


class SERPCollector:
    def __init__(self, enable_spell_check=True):
        # импортируем основной класс
        self.search_engine = HabrSearchEngine(enable_spell_check)

    def collect_serp_data(self, queries_config, output_json="serp_data.json", output_xlsx="serp_data.xlsx"):
        """Собирает данные по запросам и сохраняет в JSON и XLSX"""

        all_serp_data = {}
        all_articles = []

        for i, config in enumerate(queries_config, 1):
            query = config['query']
            search_type = config['type']

            print(f"\n[{i}/{len(queries_config)}] Запрос: '{query}' (тип: {search_type})")

            # Используем существующий метод search_articles из основного класса
            results = self.search_engine.search_articles(query, size=10, search_type=search_type)

            if not results or 'hits' not in results:
                print(f"  ⚠️  Не удалось получить результаты для запроса: {query}")
                continue

            hits = results['hits']['hits']
            query_results = []

            for rank, hit in enumerate(hits, 1):
                source = hit['_source']

                ml_score = hit.get('_ml_score', 0)  # ← используем get для безопасности
                combined_score = hit.get('_combined_score', 0)  # ← берем значение из hit

                article_data = {
                    'relevance': None,  # ПЕРВЫЙ СТОЛБЕЦ - для ручной разметки
                    'query_id': i,
                    'query_text': query,
                    'search_type': search_type,
                    'rank': rank,
                    'score': float(hit['_score']),
                    'ml_score': ml_score,
                    'combined_score': combined_score,
                    'title': source['title'],
                    'url': source['url'],
                    'author': source.get('author', 'Неизвестен'),
                    'date': source.get('date', ''),
                    'hubs': ', '.join(source.get('hubs', [])),
                    'tags': ', '.join(source.get('tags', [])),
                }

                query_results.append(article_data)
                all_articles.append(article_data)

                print(f"  {rank}. _score=[{hit['_score']:.2f}] _ml_score={ml_score:.2f} _combined_score={combined_score:.2f} {source['title'][:70]}...")

            all_serp_data[f"{query}_{search_type}"] = {
                'search_type': search_type,
                'total_found': results['hits']['total']['value'],
                'articles_fetched': len(hits),
                'results': query_results
            }

        # Сохраняем в JSON
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(all_serp_data, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n✅ Данные сохранены в JSON: {output_json}")

        # Сохраняем в XLSX
        if all_articles:
            # Создаем DataFrame с правильным порядком колонок
            columns_order = [
                'relevance',  # ПЕРВЫЙ СТОЛБЕЦ
                'query_id',
                'query_text',
                'search_type',
                'rank',
                'score',
                'ml_score',
                'combined_score',
                'title',
                'url',
                'author',
                'date',
                'hubs',
                'tags'
            ]

            df = pd.DataFrame(all_articles)
            df = df[columns_order]  # Упорядочиваем колонки

            # Сохраняем в Excel
            with pd.ExcelWriter(output_xlsx, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='SERP_Data', index=False)

            print(f"✅ Данные сохранены в XLSX: {output_xlsx}")

            # Статистика
            total_articles = len(all_articles)
            unique_queries = len(queries_config)
            print(f"\nСтатистика:")
            print(f"   Всего запросов: {unique_queries}")
            print(f"   Всего статей: {total_articles}")
            print(f"   Статей на запрос: {total_articles / unique_queries:.1f}")

        return all_serp_data


def main():
    """Основная функция для запуска сбора SERP данных"""
    try:
        print("🔍 Сбор SERP данных для 10 тестовых запросов")
        print("=" * 60)
        print("Конфигурация запросов:")
        print("  10 запросов с SIMPLE поиском (минимум 67% слов)")
        print("=" * 60)
        print("Всего будет собрано: 10 запросов × 10 статей = 100 статей")
        print("=" * 60)

        # Создаем коллектор
        collector = SERPCollector()

        # Собираем данные
        serp_data = collector.collect_serp_data(
            queries_config=test_queries_config,
            output_json="habr_serp_data.json",
            output_xlsx="habr_serp_data.xlsx"
        )

        print("\nСбор данных завершен успешно!")
        print("\nСозданные файлы:")
        print("  - habr_serp_data.json - структурированные данные SERP")
        print("  - habr_serp_data.xlsx - табличные данные для анализа (Excel)")

        # Дополнительная информация о собранных данных
        if serp_data:
            total_queries = len(serp_data)
            total_articles = sum(len(data['results']) for data in serp_data.values())

            print(f"\nИтоговая статистика:")
            print(f"  Успешно обработано запросов: {total_queries}/10")
            print(f"  Всего собрано статей: {total_articles}")

            print("\nСледующие шаги:")
            print("  1. Откройте файл habr_serp_data.xlsx в Excel")
            print("  2. В столбце 'relevance' (ПЕРВЫЙ СТОЛБЕЦ) поставьте:")
            print("     - 1 для релевантных статей")
            print("     - 0 для нерелевантных статей")
            print("  3. Сохраните файл после разметки")

    except Exception as e:
        print(f"\nПроизошла ошибка при сборе данных: {e}")
        print("Проверьте:")
        print("  - Запущен ли Elasticsearch на localhost:9200")
        print("  - Корректность импорта класса HabrSearchEngine")
        print("  - Наличие индекса 'habr_articles' в Elasticsearch")
        print("  - Установлены ли библиотеки: pandas, openpyxl")


if __name__ == "__main__":
    main()