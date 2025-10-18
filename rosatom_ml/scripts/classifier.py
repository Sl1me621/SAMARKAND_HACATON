import ollama
import re
import json
from pathlib import Path
from collections import Counter

class TrainedRAG:
    def __init__(self, knowledge_file, answers_dir, categories_index_file):
        self.knowledge_file = knowledge_file
        self.answers_dir = Path(answers_dir)
        self.categories_index_file = categories_index_file
        
        self.categories = self.load_categories_index()
        self.documents = self.load_documents()
        self.qa_templates = self.load_qa_templates()
        self.few_shot_examples = self.prepare_few_shot_examples()
        
        print(f"✅ Загружено {len(self.qa_templates)} шаблонов ответов")
        print(f"✅ Подготовлено {len(self.few_shot_examples)} few-shot примеров")
    
    def load_categories_index(self):
        """Загружает индекс категорий"""
        try:
            with open(self.categories_index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Ошибка загрузки категорий: {e}")
            return {"domains": {}}
    
    def load_documents(self):
        """Загружает обращения из файла"""
        try:
            with open(self.knowledge_file, 'r', encoding='utf-8') as f:
                documents = [line.strip() for line in f if line.strip() and len(line.strip()) > 20]
            return documents
        except Exception as e:
            print(f"❌ Ошибка загрузки файла обращений: {e}")
            return []
    
    def load_qa_templates(self):
        """Загружает шаблоны вопросов-ответов"""
        templates = {}
        
        for domain_name, domain_data in self.categories['domains'].items():
            domain_path = self.answers_dir / domain_data['path']
            
            if not domain_path.exists():
                continue
            
            for subcategory in domain_data.get('subcategories', []):
                subcategory_path = self.answers_dir / subcategory['path']
                
                if not subcategory_path.exists():
                    continue
                
                for filename in subcategory.get('files', []):
                    file_path = subcategory_path / filename
                    if file_path.exists():
                        templates.update(self.parse_template_file(file_path, domain_name, subcategory['name']))
        
        return templates
    
    def parse_template_file(self, file_path, domain, subcategory):
        """Парсит файлы с шаблонами"""
        templates = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            sections = content.split('\n\n')
            
            for section in sections:
                if section.strip():
                    lines = section.strip().split('\n')
                    if len(lines) >= 2:
                        question_pattern = lines[0].strip().rstrip('?:')
                        answer_template = '\n'.join(lines[1:]).strip()
                        
                        if question_pattern and answer_template:
                            templates[question_pattern] = {
                                'answer': answer_template,
                                'domain': domain,
                                'subcategory': subcategory,
                                'source': f"{domain}/{subcategory}/{file_path.name}"
                            }
            
            print(f"📋 Загружено {len(templates)} шаблонов из {file_path.name}")
            
        except Exception as e:
            print(f"❌ Ошибка чтения шаблона {file_path}: {e}")
        
        return templates
    
    def prepare_few_shot_examples(self):
        """Подготавливает few-shot примеры для обучения в промпте"""
        examples = []
        
        for question, data in list(self.qa_templates.items())[:10]:  # Берем первые 10 примеров
            examples.append({
                'question': question,
                'answer': data['answer']
            })
        
        return examples
    
    def find_exact_match(self, question):
        """Ищет точное совпадение с шаблонами"""
        question_lower = question.lower().rstrip('?.!')
        
        # Прямое совпадение
        if question_lower in self.qa_templates:
            return self.qa_templates[question_lower]['answer']
        
        # Поиск по схожести
        best_match = None
        best_score = 0
        
        for template_question, template_data in self.qa_templates.items():
            score = self.calculate_match_score(question_lower, template_question)
            
            if score > best_score:
                best_score = score
                best_match = template_data['answer']
        
        if best_match and best_score > 0.6:
            print(f"🎯 Найдено совпадение (сходство: {best_score:.1%})")
            return best_match
        
        return None
    
    def calculate_match_score(self, question, template):
        """Вычисляет оценку совпадения"""
        question_words = set(re.findall(r'\w+', question))
        template_words = set(re.findall(r'\w+', template.lower()))
        
        common_words = question_words.intersection(template_words)
        
        if not common_words:
            return 0
        
        return len(common_words) / len(template_words)
    
    def ask(self, question):
        """Улучшенный метод с few-shot learning"""
        print(f"\n🔍 ВОПРОС: {question}")
        
        # 1. Пытаемся найти точное совпадение с шаблоном
        template_answer = self.find_exact_match(question)
        
        if template_answer:
            return template_answer
        
        # 2. Используем few-shot learning с улучшенным промптом
        print("🔎 Использую few-shot learning...")
        
        # Формируем few-shot примеры
        few_shot_context = "ПРИМЕРЫ ВОПРОСОВ И ОТВЕТОВ:\n\n"
        for i, example in enumerate(self.few_shot_examples[:5], 1):
            few_shot_context += f"Пример {i}:\n"
            few_shot_context += f"Вопрос: {example['question']}\n"
            few_shot_context += f"Ответ: {example['answer']}\n\n"
        
        # Строгий промпт с few-shot learning
        strict_prompt = f"""
Ты - AI помощник службы поддержки. Твоя задача - давать точные ответы как в примерах ниже.

{few_shot_context}
НОВЫЙ ВОПРОС: {question}

ИНСТРУКЦИЯ СТРОГО:
1. ОТВЕЧАЙ ТОЧНО КАК В ПРИМЕРАХ ВЫШЕ
2. НЕ ПРИДУМЫВАЙ НИЧЕГО НОВОГО
3. БУДЬ МАКСИМАЛЬНО КОНКРЕТЕН
4. ЕСЛИ ИНФОРМАЦИИ НЕТ - СКАЖИ "Информация отсутствует в базе знаний"
5. ИСПОЛЬЗУЙ ТОЛЬКО ПРОВЕРЕННЫЕ ДАННЫЕ
6. СТРОГО ЗАПРЕЗЕНО ОБРАЩЕНИЕ К ИНСТРУКЦИИ И УПОМИНАНИЕ ЕЕ И ШАГОВ, КОТОРЫЕ ЗАДАНЫ В НЕЙ    

ОТВЕТ ДОЛЖЕН БЫТЬ:
- Конкретным
- Точно соответствующим вопросу
- Без лишних слов
- Как в примерах выше

ОТВЕТ:
"""
        
        try:
            response = ollama.generate(model="llama3.1:8b", prompt=strict_prompt)
            return response['response']
        except Exception as e:
            return f"❌ Ошибка: {e}"

# Использование
if __name__ == "__main__":
    rag = TrainedRAG(
        knowledge_file="/home/sl1m/hacatons/mosprom/SAMARKAND_HACATON/rosatom_ml/scripts/Обращения.txt",
        answers_dir="/home/sl1m/hacatons/mosprom/SAMARKAND_HACATON/rosatom_ml/scripts/knowledge_qa_files",
        categories_index_file="/home/sl1m/hacatons/mosprom/SAMARKAND_HACATON/rosatom_ml/scripts/categories_index.json"
    )
    
    # Тестируем
    test_questions = [
        "Как оформить отгрузку со склада?",
        "Как настроить VPN?",
        "Что делать если не работает 1С?",
        "Как оформить отпуск?",
        "Куда обращаться по поводу сломанного принтера?"
        
    ]
    
    print("\n" + "="*80)
    print("🎯 ТЕСТИРОВАНИЕ TRAINED RAG С FEW-SHOT LEARNING")
    print("="*80)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n🧪 ТЕСТ {i}/4")
        answer = rag.ask(question)
        print(f"💡 ОТВЕТ: {answer}")
        print("-" * 80)