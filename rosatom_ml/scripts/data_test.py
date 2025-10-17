# data_processing.py
import pandas as pd
from model.optimized_llama import OptimizedLlamaClassifier

def process_real_tickets():
    """Обработка реальных обращений с Llama"""
    classifier = OptimizedLlamaClassifier()
    
    # Чтение твоих данных
    with open('Обращения.txt', 'r', encoding='utf-8') as f:
        tickets = [line.strip() for line in f if line.strip()]
    
    results = []
    print("Обрабатываю обращения с Llama...")
    
    for i, ticket in enumerate(tickets[:50]):  # Первые 50 для теста
        if len(ticket) > 20:  # Только значимые обращения
            try:
                result = classifier.classify_text(ticket)
                results.append({
                    'text': ticket,
                    'category': result['category'],
                    'confidence': result['confidence']
                })
                print(f"{i+1}/{min(50, len(tickets))}: {result['category']} - {ticket[:60]}...")
            except Exception as e:
                print(f"Ошибка с обращением {i}: {e}")
                continue
    
    # Сохраняем результаты
    df = pd.DataFrame(results)
    df.to_csv('llama_classified_tickets.csv', index=False, encoding='utf-8')
    print(f"\nОбработано {len(results)} обращений")
    print(df['category'].value_counts())
    
    return df

# Быстрый тест
if __name__ == "__main__":
    process_real_tickets()