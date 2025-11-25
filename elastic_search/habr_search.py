from elasticsearch import Elasticsearch
import logging
from datetime import datetime
import requests
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Добавляем импорт ML-ранкера
try:
    from ml_ranker import MLRelevanceRanker
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("ML-ранкинг недоступен - установите необходимые зависимости")

# Добавляем импорт предсказателя тегов
try:
    from ml.tag_predictor import TagPredictor
    TAG_PREDICTION_AVAILABLE = True
except ImportError:
    TAG_PREDICTION_AVAILABLE = False
    print("Предсказание тегов недоступно")

class HabrSearchEngine:
    def __init__(self, enable_spell_check=True, enable_ml_ranking=True, enable_tag_prediction=True): # enable_spell_check=True - спрашивает у пользователя про опечатку, False - не проверяет на опечатки
        self.es = Elasticsearch(["http://localhost:9200"])
        self.enable_spell_check = enable_spell_check
        self.enable_ml_ranking = enable_ml_ranking
        self.enable_tag_prediction = enable_tag_prediction

        # Инициализируем ML-ранкер если доступен
        if enable_ml_ranking and ML_AVAILABLE:
            try:
                self.ml_ranker = MLRelevanceRanker()
                print("ML-ранкинг активирован")
            except Exception as e:
                print(f"ML-ранкинг недоступен: {e}")
                self.enable_ml_ranking = False
        else:
            self.enable_ml_ranking = False

        # Инициализируем предсказатель тегов если доступен
        if enable_tag_prediction and TAG_PREDICTION_AVAILABLE:
            try:
                self.tag_predictor = TagPredictor()
                print("AI-теги активированы")
            except Exception as e:
                print(f"AI-теги недоступны: {e}")
                self.enable_tag_prediction = False
        else:
            self.enable_tag_prediction = False

        if not self.es.ping():
            raise ConnectionError("Не удалось подключиться к Elasticsearch")

    def smart_spell_check(self, query):
        """Умная проверка орфографии через Yandex Speller API"""
        try:
            # Yandex Speller API
            url = "https://speller.yandex.net/services/spellservice.json/checkText"
            params = {
                'text': query,
                'lang': 'ru,en',  # Русский и английский
                'options': 518  # Игнорировать цифры, URLs и т.д.
            }

            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                corrections = response.json()

                if corrections:
                    corrected_query = query
                    corrections_list = []

                    for correction in corrections:
                        if correction.get('s'):
                            corrected_word = correction['s'][0]
                            original_word = correction['word']
                            corrected_query = corrected_query.replace(original_word, corrected_word)
                            corrections_list.append(f"'{original_word}' → '{corrected_word}'")

                    if corrected_query != query:
                        print(f"\nНайдены возможные опечатки:")
                        for correction in corrections_list:
                            print(f"   {correction}")

                        while True:
                            choice = input(f"\nИсправить на '{corrected_query}'? (y/n): ").strip().lower()
                            if choice in ['y', 'yes', 'да', 'д']:
                                return corrected_query
                            elif choice in ['n', 'no', 'нет', 'н']:
                                return query
                            else:
                                print("Пожалуйста, введите 'y' (да) или 'n' (нет)")

            return query

        except Exception as e:
            logger.warning(f"Ошибка проверки орфографии: {e}")
            return query

    def is_exact_phrase(self, query):
        """Проверяет, является ли запрос точной фразой в кавычках"""
        # Проверяем разные типы кавычек
        quote_pairs = [
            ('"', '"'),
            ('«', '»'),
            ('“', '”'),
            ("'", "'")
        ]

        for open_quote, close_quote in quote_pairs:
            if query.startswith(open_quote) and query.endswith(close_quote):
                return True

        return False

    def extract_phrase_from_quotes(self, query):
        """Извлекает фразу из кавычек"""
        if query.startswith('"') and query.endswith('"'):
            return query[1:-1]
        elif query.startswith('«') and query.endswith('»'):
            return query[1:-1]
        elif query.startswith('“') and query.endswith('”'):
            return query[1:-1]
        elif query.startswith("'") and query.endswith("'"):
            return query[1:-1]
        return query

    def should_use_spell_check(self, query):
        """Определяем, нужно ли использовать проверку орфографии"""
        if self.enable_spell_check == False:
            return False

        # Не проверяем точные фразы в кавычках
        if self.is_exact_phrase(query):
            print("Используется поиск точной фразы (без исправлений)")
            return False

        return True

    def predict_article_tags(self, title, text):
        if not self.enable_tag_prediction or not hasattr(self, 'tag_predictor'):
            return []
        return self.tag_predictor.predict_tags(title, text)

    def search_articles(self, query, size=10, search_type="simple", use_ml_ranking=None):
        """Поиск статей с различными типами запросов с возможностью ML-ранжирования"""

        # Определяем использовать ли ML-ранжирование
        if use_ml_ranking is None:
            use_ml_ranking = self.enable_ml_ranking

        # Обрабатываем точные фразы в кавычках
        if self.is_exact_phrase(query):
            exact_phrase = self.extract_phrase_from_quotes(query)
            search_body = {
                "query": {
                    "match_phrase": {
                        "text": {
                            "query": exact_phrase,
                            "slop": 2  # Допускаем небольшое расстояние между словами
                        }
                    }
                },
                "highlight": {
                    "fields": {
                        "title": {},
                        "text": {"fragment_size": 150, "number_of_fragments": 3},
                        "hubs": {},
                        "tags": {}
                    }
                }
            }

            try:
                response = self.es.search(
                    index="habr_articles",
                    body=search_body,
                    size=size
                )

                # Применяем ML-ранжирование если включено
                if use_ml_ranking and ML_AVAILABLE:
                    response = self.ml_ranker.rerank_results(query, response)

                return response
            except Exception as e:
                logger.error(f"Ошибка поиска точной фразы: {e}")
                return None

        # Умная проверка орфографии
        if self.should_use_spell_check(query):
            corrected_query = self.smart_spell_check(query)
            if corrected_query != query:
                query = corrected_query

        if search_type == "exact":
            # ТОЧНЫЙ поиск - должны присутствовать все слова (оператор AND)
            search_body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^3", "text^2", "hubs^2", "tags^2", "author"],
                        "operator": "and"  # Все слова должны быть, все 100%
                    }
                },
                "highlight": {
                    "fields": {
                        "title": {},
                        "text": {"fragment_size": 150, "number_of_fragments": 3},
                        "hubs": {},
                        "tags": {}
                    }
                }
            }
        elif search_type == "simple":
            # Оператор OR
            search_body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^5", "text^2", "tags^2", "hubs^2", "author"],
                        "operator": "or",
                        "minimum_should_match": "67%"
                    }
                },
                "highlight": {
                    "fields": {
                        "title": {},
                        "text": {"fragment_size": 150, "number_of_fragments": 3},
                        "hubs": {},
                        "tags": {}
                    }
                }
            }
        elif search_type == "boost":
            # Экспериментальный поиск с бустингом
            search_body = {
                "query": {
                    "bool": {
                        "should": [
                            {
                                "match": {
                                    "title": {
                                        "query": query,
                                        "boost": 3
                                    }
                                }
                            },
                            {
                                "match": {
                                    "text": {
                                        "query": query,
                                        "boost": 2
                                    }
                                }
                            },
                            {
                                "match": {
                                    "hubs": {
                                        "query": query,
                                        "boost": 2
                                    }
                                }
                            },
                            {
                                "match": {
                                    "tags": {
                                        "query": query,
                                        "boost": 2
                                    }
                                }
                            }
                        ],
                        "minimum_should_match": "67%"
                    }
                },
                "highlight": {
                    "fields": {
                        "title": {},
                        "text": {"fragment_size": 150, "number_of_fragments": 3},
                        "hubs": {},
                        "tags": {}
                    }
                }
            }

        try:
            response = self.es.search(
                index="habr_articles",
                body=search_body,
                size=size
            )

            # Применяем ML-ранжирование если включено
            if use_ml_ranking and ML_AVAILABLE:
                response = self.ml_ranker.rerank_results(query, response)

            return response
        except Exception as e:
            logger.error(f"Ошибка поиска: {e}")
            return None

    def format_search_results(self, results):
        """Форматирует результаты поиска с учетом ML-ранжирования"""

        if not results or 'hits' not in results or 'hits' not in results['hits']:
            print("Ничего не найдено")
            print("Попробуйте:")
            print("   - Изменить запрос")
            print("   - Использовать другой режим поиска")
            print("   - Проверить орфографию")
            return

        total = results['hits']['total']['value']
        hits = results['hits']['hits']

        print(f"\nНайдено результатов: {total}")

        # Показываем тип ранжирования
        features = []
        if hasattr(self, 'enable_ml_ranking') and self.enable_ml_ranking:
            features.append("ML-ранжирование")
        if hasattr(self, 'enable_tag_prediction') and self.enable_tag_prediction:
            features.append("AI-теги")

        if features:
            print(f"Функции: {', '.join(features)}")
        else:
            print("Ранжирование: стандартное ElasticSearch")

        print("=" * 80)

        for i, hit in enumerate(hits, 1):
            source = hit['_source']
            # score = hit['_score']

            # Показываем разные скоринги в зависимости от типа ранжирования
            if '_combined_score' in hit:
                # ML-ранжирование
                score_display = f"{hit['_combined_score']:.3f} (ML)"
                ml_info = f", ML: {hit['_ml_score']:.3f}"
            else:
                # Стандартное ранжирование
                score_display = f"{hit['_score']:.2f}"
                ml_info = ""

            print(f"\n{i}. [{score_display}] {source['title']}")
            print(f"   Автор: {source.get('author', 'Неизвестен')}")

            if 'date' in source:
                try:
                    date = datetime.fromisoformat(source['date'].replace('Z', '+00:00'))
                    print(f"   Дата: {date.strftime('%d.%m.%Y %H:%M')}")
                except:
                    print(f"   Дата: {source['date']}")

            # Выводим хабы и теги
            if source.get('hubs'):
                print(f"   Хабы: {', '.join(source['hubs'][:3])}")
            if source.get('tags'):
                print(f"   Теги: {', '.join(source['tags'][:3])}")


            # AI-предсказанные теги
            if self.enable_tag_prediction:
                text = source.get('text', '')[:1000] # :1000 - опционально : можно поиграться с этим параметром
                predicted_tags = self.predict_article_tags(source['title'], text)
                if predicted_tags:
                    tags_str = ', '.join([f"{tag['tag']} ({tag['confidence']})" for tag in predicted_tags[:3]])
                    print(f"AI-теги: {tags_str}")

            # Дополнительная информация о скоринге
            if ml_info:
                print(f"   Скоринг: ES: {hit['_score']:.2f}{ml_info}")


            # Выводим подсветку
            if 'highlight' in hit:
                if 'title' in hit['highlight']:
                    highlighted = hit['highlight']['title'][0].replace('<em>', '**').replace('</em>', '**')
                    print(f"   Заголовок: ...{highlighted}...")
                if 'text' in hit['highlight']:
                    for fragment in hit['highlight']['text'][:2]:
                        cleaned = fragment.replace('<em>', '**').replace('</em>', '**')
                        print(f"   Текст: ...{cleaned}...")

            print(f"   URL: {source['url']}")
            print("-" * 80)


def main():
    try:
        search_engine = HabrSearchEngine(enable_ml_ranking=True, enable_tag_prediction=True)
        print("Умная поисковая система Habr с ML-ранжированием")
        print("Доступные команды:")
        print("  /exact   - точный поиск (все 100% слов)")
        print("  /simple  - простой поиск (для +2 слов минимум 75% из них)")
        print("  /boost - поиск с бустингом (для +2 слов минимум 75% из них)")
        print("  /ml_on   - включить ML-ранжирование")
        print("  /ml_off  - выключить ML-ранжирование")
        print("  /tags_on - включить AI-теги")
        print("  /tags_off - выключить AI-теги")
        print("  /exit    - выход")

        print("\nОсобенности:")
        print("  - ML-ранжирование на основе обученной модели")
        print("  - Интерактивное исправление опечаток")
        print("  - AI-предсказание тегов по содержанию")
        print("  - Поиск точных фраз в кавычках")
        print("  - Умные подсказки")

        print("\nПримеры:")
        print("  /exact русы против ящеров")
        print("  /simple пайтон машинное обyчение")
        print("  /boost база данных")
        print('  "точная фраза в кавычках"')
        print('  «русские кавычки тоже работают»')

        ml_enabled = True
        tags_enabled = True

        while True:
            try:
                user_input = input("\nВведите запрос: ").strip()

                if user_input.lower() in ['/exit', 'exit', 'quit']:
                    break
                elif user_input == '/ml_on':
                    search_engine.enable_ml_ranking = True
                    ml_enabled = True
                    print("ML-ранжирование включено")
                    continue
                elif user_input == '/ml_off':
                    search_engine.enable_ml_ranking = False
                    ml_enabled = False
                    print("ML-ранжирование выключено")
                    continue
                elif user_input == '/tags_on':
                    search_engine.enable_tag_prediction = True
                    tags_enabled = True
                    print("AI-теги включены")
                    continue
                elif user_input == '/tags_off':
                    search_engine.enable_tag_prediction = False
                    tags_enabled = False
                    print("AI-теги выключены")
                    continue
                elif user_input.startswith('/exact '):
                    query = user_input[7:]
                    search_type = "exact"
                elif user_input.startswith('/simple '):
                    query = user_input[8:]
                    search_type = "simple"
                elif user_input.startswith('/boost '):
                    query = user_input[7:]
                    search_type = "boost"
                else:
                    # По умолчанию используем простой поиск
                    query = user_input
                    search_type = "simple"

                if not query:
                    continue

                features = []
                if ml_enabled:
                    features.append("ML")
                if tags_enabled:
                    features.append("AI-теги")

                features_str = f" ({', '.join(features)})" if features else ""

                print(f"\nПоиск: '{query}' (режим: {search_type}{features_str})")

                results = search_engine.search_articles(query, size=10, search_type=search_type)
                search_engine.format_search_results(results)

            except KeyboardInterrupt:
                print("\nВыход...")
                break
            except Exception as e:
                logger.error(f"Ошибка: {e}")

    except ConnectionError as e:
        print(f"Ошибка: {e}")
        print("Убедитесь, что Elasticsearch запущен на localhost:9200")


if __name__ == "__main__":
    main()