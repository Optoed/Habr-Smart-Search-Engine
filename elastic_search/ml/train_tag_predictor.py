import pandas as pd
from tag_predictor import train_tag_predictor


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
        print("Доступные колонки:", list(df.columns))
        return

    # 3. Проверяем, что есть данные с тегами
    articles_with_tags = df['tags'].notna().sum()
    print(f"Статьи с тегами: {articles_with_tags}/{len(df)}")

    if articles_with_tags < 10:
        print("Слишком мало статей с тегами для обучения")
        return

    # 4. Обучаем модель
    print("Запуск обучения модели...")
    train_tag_predictor(df)

    print("\nОбучение завершено!")


if __name__ == "__main__":
    main()