import pandas as pd
import numpy as np


def calculate_metrics(excel_file):
    # Загружаем данные
    df = pd.read_excel(excel_file)

    print("РАСЧЕТ МЕТРИК КАЧЕСТВА ПОИСКА")
    print("=" * 50)

    # 1. PRECISION@5 - точность в топ-5 результатах
    def precision_at_5(df):
        """Precision@5 - доля релевантных документов в топ-5"""
        precisions = []
        for query in df['query_text'].unique():
            query_data = df[df['query_text'] == query].head(5)
            relevant_count = len(query_data[query_data['relevance'] == 1])
            precision = relevant_count / 5
            precisions.append(precision)
        return np.mean(precisions)

    # 2. PRECISION@10 - точность в топ-10 результатах
    def precision_at_10(df):
        """Precision@10 - доля релевантных документов в топ-10"""
        precisions = []
        for query in df['query_text'].unique():
            query_data = df[df['query_text'] == query].head(10)
            relevant_count = len(query_data[query_data['relevance'] == 1])
            precision = relevant_count / 10
            precisions.append(precision)
        return np.mean(precisions)

    # 3. MRR (Mean Reciprocal Rank) - среднее обратное ранга первого релевантного
    def calculate_mrr(df):
        """MRR - среднее обратное ранга первого релевантного документа"""
        reciprocal_ranks = []
        for query in df['query_text'].unique():
            query_data = df[df['query_text'] == query]
            first_relevant = query_data[query_data['relevance'] == 1].head(1)

            if len(first_relevant) > 0:
                rank = first_relevant.iloc[0]['rank']
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0)
        return np.mean(reciprocal_ranks)

    # 4. MAP (Mean Average Precision) - средняя точность
    def calculate_map(df):
        """MAP - средняя точность по всем запросам"""
        average_precisions = []

        for query in df['query_text'].unique():
            query_data = df[df['query_text'] == query].sort_values('rank')
            relevant_docs = query_data[query_data['relevance'] == 1]

            if len(relevant_docs) == 0:
                average_precisions.append(0)
                continue

            precision_sum = 0
            for i, (idx, row) in enumerate(relevant_docs.iterrows(), 1):
                # Precision на позиции k
                docs_up_to_k = query_data[query_data['rank'] <= row['rank']]
                relevant_up_to_k = len(docs_up_to_k[docs_up_to_k['relevance'] == 1])
                precision_at_k = relevant_up_to_k / row['rank']
                precision_sum += precision_at_k

            average_precisions.append(precision_sum / len(relevant_docs))

        return np.mean(average_precisions)

    # Рассчитываем метрики
    metrics = {
        'Precision@5': precision_at_5(df),
        'Precision@10': precision_at_10(df),
        'MRR': calculate_mrr(df),
        'MAP': calculate_map(df)
    }

    return metrics, df


def print_detailed_analysis(df, metrics):
    """Детальный анализ по каждому запросу"""
    print("\nДЕТАЛЬНЫЙ АНАЛИЗ ПО ЗАПРОСАМ:")
    print("=" * 80)
    print(f"{'Запрос':<25} {'P@5':<6} {'P@10':<6} {'MRR':<6} {'Релевантных':<12} {'Всего':<6}")
    print("-" * 80)

    for query in df['query_text'].unique():
        query_data = df[df['query_text'] == query]

        # Precision@5
        top_5 = query_data.head(5)
        p5 = len(top_5[top_5['relevance'] == 1]) / 5

        # Precision@10
        top_10 = query_data.head(10)
        p10 = len(top_10[top_10['relevance'] == 1]) / 10

        # MRR для этого запроса
        first_relevant = query_data[query_data['relevance'] == 1].head(1)
        if len(first_relevant) > 0:
            mrr = 1.0 / first_relevant.iloc[0]['rank']
        else:
            mrr = 0

        # Всего релевантных
        all_relevant = len(df[(df['query_text'] == query)])
        found_relevant = len(query_data[query_data['relevance'] == 1])

        print(f"{query[:23]:<25} {p5:.3f}  {p10:.3f}  {mrr:.3f}  {found_relevant}/{all_relevant:<11} {len(query_data)}")


def save_metrics_to_excel(metrics, df, output_file="search_metrics.xlsx"):
    """Сохраняет метрики в Excel файл"""

    # Метрики по запросам
    queries_metrics = []
    for query in df['query_text'].unique():
        query_data = df[df['query_text'] == query]

        # Precision@5
        top_5 = query_data.head(5)
        p5 = len(top_5[top_5['relevance'] == 1]) / 5

        # Precision@10
        top_10 = query_data.head(10)
        p10 = len(top_10[top_10['relevance'] == 1]) / 10

        # MRR
        first_relevant = query_data[query_data['relevance'] == 1].head(1)
        if len(first_relevant) > 0:
            mrr = 1.0 / first_relevant.iloc[0]['rank']
        else:
            mrr = 0

        # Всего релевантных
        all_relevant = len(df[(df['query_text'] == query) & (df['relevance'] == 1)])
        found_relevant = len(query_data[query_data['relevance'] == 1])

        queries_metrics.append({
            'query': query,
            'precision@5': p5,
            'precision@10': p10,
            'mrr': mrr,
            'relevant_found': found_relevant,
            'total_relevant': all_relevant,
            'coverage': f"{found_relevant}/{all_relevant}"
        })

    queries_df = pd.DataFrame(queries_metrics)
    overall_df = pd.DataFrame([metrics])

    # Сохраняем в Excel
    with pd.ExcelWriter(output_file) as writer:
        overall_df.to_excel(writer, sheet_name='Общие метрики', index=False)
        queries_df.to_excel(writer, sheet_name='Метрики по запросам', index=False)

    print(f"\nОтчет сохранен в {output_file}")


# Основная программа
if __name__ == "__main__":
    excel_file = "habr_serp_data.xlsx"

    try:
        metrics, df = calculate_metrics(excel_file)

        # Выводим общие метрики
        print("\nОСНОВНЫЕ МЕТРИКИ КАЧЕСТВА:")
        print("-" * 35)
        for metric, value in metrics.items():
            print(f"{metric:<12}: {value:.3f}")

        # Детальный анализ
        print_detailed_analysis(df, metrics)

        # Сохраняем в Excel
        save_metrics_to_excel(metrics, df, "search_metrics_after_ml.xlsx")

        # Интерпретация результатов
        print("\nИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ:")
        print(f"• Precision@5: {metrics['Precision@5']:.1%} - в среднем {metrics['Precision@5'] * 5:.1f} из 5 топ-результатов релевантны")
        print(f"• Precision@10: {metrics['Precision@10']:.1%} - в среднем {metrics['Precision@10'] * 10:.1f} из 10 результатов релевантны")
        print(f"• MRR: {metrics['MRR']:.3f} - первый релевантный результат в среднем на позиции {1/max(metrics['MRR'], 0.001):.1f}")
        print(f"• MAP: {metrics['MAP']:.3f} - общее качество ранжирования (чем ближе к 1, тем лучше)")

    except FileNotFoundError:
        print(f"Файл {excel_file} не найден!")
    except Exception as e:
        print(f"Ошибка: {e}")