import pandas as pd
from dotenv import load_dotenv
import requests
import os
import time

# Загружаем переменные из .env файла
load_dotenv()

api_key = os.getenv("API_KEY")
ai_model = "mistralai/Magistral-Small-2506"

xlsx_path_from = "all_serp_data_for_ml.xlsx"
xlsx_path_to = "all_serp_data_for_ml.xlsx"

def make_request(message_text):
    url = "https://api.intelligence.io.solutions/api/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_key
    }

    data = {
        "model": ai_model,
        "messages": [
            {
                "role": "system",
                "content": """Ты оцениваешь релевантность выданных по запросу статей.
                 Отвечай ТОЛЬКО цифрами 1 (если статья более менее соответствует поисковому запросу или хотя бы теме) или 0 (если статья СОВСЕМ не соответствует поисковому запросу) через запятую. 
                 Ставь 0, только если результаты поиска и близко не соответствуют запросу.
                 Релевантный пример query = python машинное обучение, title = Внедрение ML кластера для масштабирования AI сервисов, tags = Теги:MLмасштабированиеdockerkubernetesaiархитектура по, ML, масштабирование, docker, kubernetes, ai, архитектура по
                 Нерелевантный пример query = python машинное обучение, title = Есть ли у AMD перспективы в AI/ML/DL. Часть 1, tags = Теги:selectelmlaidlamdnvidiaмашинное обучение, selectel, ml, ai, dl, amd, nvidia, машинное обучение
                 """,
            },
            {
                "role": "user",
                "content": message_text
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    data = response.json()

    if 'choices' in data and len(data['choices']) > 0:
        return data['choices'][0]['message']['content']
    else:
        return data


def main():
    # 1) Открываем xlsx
    df = pd.read_excel(xlsx_path_from)

    print(f"Всего строк в файле: {len(df)}")

    # Получаем уникальные query_id
    query_ids = df['query_id'].unique()

    total_processed = 0

    # 2) Проходим по всем запросам
    for query_id in query_ids:
        print(f"\n{'=' * 50}")
        print(f"Обрабатываем запрос ID: {query_id}")

        # Получаем все статьи для этого запроса
        query_articles = df[df['query_id'] == query_id]
        query_text = query_articles['query_text'].iloc[0]
        num_articles = len(query_articles)

        print(f"Запрос: '{query_text}'")
        print(f"Количество статей: {num_articles}")

        # Проверяем, есть ли уже обработанные статьи в этой группе
        already_processed = query_articles['relevance'].notna().sum()
        if already_processed == num_articles:
            print(f"Все {num_articles} статей уже обработаны, пропускаем")
            total_processed += num_articles
            continue
        elif already_processed > 0:
            print(f"Частично обработано: {already_processed}/{num_articles} статей")

        # 3) Формируем промпт
        prompt = f"Запрос: {query_text}\n\nОцени релевантность статей относительно запроса (1 - релевантна, 0 - не релевантна):\n\n"

        for i, (idx, article) in enumerate(query_articles.iterrows(), 1):
            title = article['title']
            tags = article['tags'][:250]
            prompt += f"{i}. title={title}. tags={tags}\n"


        prompt += f"\nОтвет (только {num_articles} цифр через запятую):"

        print("Отправляем запрос к модели...")

        try:
            # 4) Получаем ответ
            response = make_request(prompt)
            print(f"Ответ модели: {response}")

            # 5) Парсим ответ
            ratings = []
            for char in response:
                if char in '01':
                    ratings.append(int(char))
                if len(ratings) == num_articles:
                    break

            print(f"Спарсенные оценки: {ratings}")

            # 6) Сохраняем результаты в DataFrame
            if len(ratings) == num_articles:
                for i, (idx, article) in enumerate(query_articles.iterrows()):
                    df.at[idx, 'relevance'] = ratings[i]

                total_processed += num_articles
                print(f"Обработано статей: {num_articles} | Всего: {total_processed}")

                # Сохраняем прогресс после каждого запроса
                df.to_excel(xlsx_path_to, index=False)
                print("Прогресс сохранен")
            else:
                print(f"Ошибка: получено {len(ratings)} оценок, ожидалось {num_articles}")

            # Пауза между запросами
            # time.sleep(1)

        except Exception as e:
            print(f"Ошибка при обработке запроса {query_id}: {e}")

    print(f"\nОБРАБОТКА ЗАВЕРШЕНА!")
    print(f"Результаты сохранены в: {xlsx_path_to}")


if __name__ == "__main__":
    main()