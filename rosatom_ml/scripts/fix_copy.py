import ollama
import re
import json
import datetime
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict

class TrainedRAGWithExactCopy:
    def __init__(self, knowledge_file, answers_dir, categories_index_file, logs_dir="logs"):
        self.knowledge_file = knowledge_file
        self.answers_dir = Path(answers_dir)
        self.categories_index_file = categories_index_file
        self.logs_dir = Path(logs_dir)
        
        # Создаем папку для логов
        self.logs_dir.mkdir(exist_ok=True)
        
        # Загружаем данные
        self.categories = self.load_categories_index()
        self.documents = self.load_documents()
        self.qa_templates = self.load_qa_templates()
        self.few_shot_examples = self.prepare_few_shot_examples()
        
        # Загружаем историю логов
        self.interaction_logs = self.load_interaction_logs()
        self.feedback_data = self.load_feedback_data()
        
        print(f"✅ Загружено {len(self.qa_templates)} шаблонов ответов")
        print(f"✅ Подготовлено {len(self.few_shot_examples)} few-shot примеров")
        print(f"📊 Загружено {len(self.interaction_logs)} исторических записей")
        
        # ДЕБАГ: Покажем какие вопросы загружены
        print(f"\n🔍 ДЕБАГ: Первые 5 загруженных вопросов:")
        for i, (q, data) in enumerate(list(self.qa_templates.items())[:5]):
            print(f"  {i+1}. '{q}' -> {data['domain']}/{data['subcategory']}")
    
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
        """Загружает шаблоны вопросов-ответов с ТОЧНЫМ сохранением ответов"""
        templates = {}
        
        for domain_name, domain_data in self.categories['domains'].items():
            domain_path = self.answers_dir / domain_data['path']
            
            if not domain_path.exists():
                print(f"⚠️ Домен не найден: {domain_path}")
                continue
            
            for subcategory in domain_data.get('subcategories', []):
                subcategory_path = self.answers_dir / subcategory['path']
                
                if not subcategory_path.exists():
                    print(f"⚠️ Подкатегория не найдена: {subcategory_path}")
                    continue
                
                for filename in subcategory.get('files', []):
                    file_path = subcategory_path / filename
                    if file_path.exists():
                        file_templates = self.parse_template_file_exact(file_path, domain_name, subcategory['name'])
                        templates.update(file_templates)
                        print(f"📁 {domain_name}/{subcategory['name']}/{filename}: {len(file_templates)} вопросов")
                    else:
                        print(f"⚠️ Файл не найден: {file_path}")
        
        return templates
    
    def parse_template_file_exact(self, file_path, domain, subcategory):
        """Парсит файлы с ТОЧНЫМ сохранением ответов - ИСПРАВЛЕННАЯ ВЕРСИЯ"""
        templates = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"\n📖 Чтение файла: {file_path.name}")
            print(f"📄 Содержимое файла:\n{content}\n{'-'*50}")
            
            # Разные стратегии парсинга в зависимости от формата
            if "Вопрос:" in content and "Ответ:" in content:
                # Формат с явными метками "Вопрос:" и "Ответ:"
                templates.update(self.parse_with_labels(content, domain, subcategory, file_path.name))
            else:
                # Простой формат - вопрос и ответ разделены пустой строкой
                templates.update(self.parse_simple_format(content, domain, subcategory, file_path.name))
            
        except Exception as e:
            print(f"❌ Ошибка чтения шаблона {file_path}: {e}")
        
        return templates
    
    def parse_with_labels(self, content, domain, subcategory, filename):
        """Парсит файлы с явными метками 'Вопрос:' и 'Ответ:'"""
        templates = {}
        
        # Ищем блоки вопрос-ответ
        question_blocks = re.split(r'Вопрос:\s*', content)
        
        for block in question_blocks[1:]:  # Пропускаем первую часть (может быть пустой)
            if 'Ответ:' in block:
                # Разделяем на вопрос и ответ
                parts = block.split('Ответ:', 1)
                if len(parts) == 2:
                    question = parts[0].strip().rstrip('?.!')
                    answer = parts[1].strip()
                    
                    if question and answer:
                        # Создаем варианты для поиска
                        search_variants = self.generate_search_variants(question)
                        
                        for variant in search_variants:
                            templates[variant] = {
                                'exact_answer': answer,
                                'domain': domain,
                                'subcategory': subcategory,
                                'source': f"{domain}/{subcategory}/{filename}",
                                'original_question': question
                            }
                        
                        print(f"  ✅ Загружен вопрос: '{question}'")
        
        return templates
    
    def parse_simple_format(self, content, domain, subcategory, filename):
        """Парсит простой формат - вопрос и ответ разделены пустой строкой"""
        templates = {}
        
        # Разделяем на блоки по двойным переносам строк
        blocks = [b.strip() for b in content.split('\n\n') if b.strip()]
        
        for block in blocks:
            lines = [line.strip() for line in block.split('\n') if line.strip()]
            
            if len(lines) >= 2:
                # Первая строка - вопрос, остальные - ответ
                question = lines[0].rstrip('?.!')
                answer = '\n'.join(lines[1:])
                
                if question and answer:
                    # Создаем варианты для поиска
                    search_variants = self.generate_search_variants(question)
                    
                    for variant in search_variants:
                        templates[variant] = {
                            'exact_answer': answer,
                            'domain': domain,
                            'subcategory': subcategory,
                            'source': f"{domain}/{subcategory}/{filename}",
                            'original_question': question
                        }
                    
                    print(f"  ✅ Загружен вопрос: '{question}'")
        
        return templates
    
    def generate_search_variants(self, question):
        """Генерирует варианты для поиска"""
        variants = []
        base_question = question.lower().rstrip('?.!')
        
        # Основной вариант
        variants.append(base_question)
        
        # Без стоп-слов
        words = base_question.split()
        stop_words = {'как', 'что', 'где', 'когда', 'почему', 'для', 'на', 'в', 'с', 'по', 'о', 'у'}
        filtered = [w for w in words if w not in stop_words and len(w) > 2]
        if filtered:
            variants.append(' '.join(filtered))
        
        # Только ключевые слова (первые 3-4 слова)
        if len(words) > 3:
            variants.append(' '.join(words[:4]))
        
        return variants
    
    def prepare_few_shot_examples(self):
        """Подготавливает few-shot примеры для обучения в промпте"""
        examples = []
        
        for question, data in list(self.qa_templates.items())[:10]:
            examples.append({
                'question': question,
                'answer': data['exact_answer']
            })
        
        return examples
    
    # ========== СИСТЕМА ЛОГИРОВАНИЯ ==========
    
    def load_interaction_logs(self):
        """Загружает историю взаимодействий"""
        log_file = self.logs_dir / "interactions.json"
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def load_feedback_data(self):
        """Загружает данные обратной связи"""
        feedback_file = self.logs_dir / "feedback.json"
        if feedback_file.exists():
            try:
                with open(feedback_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def log_interaction(self, question, answer, used_template=False, confidence=0.0, user_feedback=None, answer_source="generated"):
        """Логирует взаимодействие с системой"""
        log_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'question': question,
            'answer': answer,
            'used_template': used_template,
            'confidence': confidence,
            'user_feedback': user_feedback,
            'answer_source': answer_source,
            'response_time': None
        }
        
        self.interaction_logs.append(log_entry)
        
        with open(self.logs_dir / "interactions.json", 'w', encoding='utf-8') as f:
            json.dump(self.interaction_logs, f, ensure_ascii=False, indent=2, default=str)
        
        self.save_interaction_to_csv(log_entry)
    
    def save_interaction_to_csv(self, log_entry):
        """Сохраняет взаимодействие в CSV файл"""
        csv_file = self.logs_dir / "interactions.csv"
        
        df_data = {
            'timestamp': [log_entry['timestamp']],
            'question': [log_entry['question']],
            'answer': [log_entry['answer']],
            'used_template': [log_entry['used_template']],
            'confidence': [log_entry['confidence']],
            'user_feedback': [log_entry['user_feedback'] or ''],
            'answer_source': [log_entry['answer_source']]
        }
        
        df = pd.DataFrame(df_data)
        
        if csv_file.exists():
            df.to_csv(csv_file, mode='a', header=False, index=False, encoding='utf-8')
        else:
            df.to_csv(csv_file, index=False, encoding='utf-8')
    
    # ========== ОСНОВНОЙ МЕТОД ВОПРОС-ОТВЕТ ==========
    
    def find_exact_match(self, question):
        """Ищет точное совпадение с шаблонами и возвращает ТОЧНЫЙ ответ"""
        question_lower = question.lower().rstrip('?.!')
        
        print(f"🔍 Поиск: '{question_lower}'")
        print(f"📚 Всего вопросов в базе: {len(self.qa_templates)}")
        
        # 1. Прямое точное совпадение
        if question_lower in self.qa_templates:
            print(f"🎯 ТОЧНОЕ СОВПАДЕНИЕ: '{question_lower}'")
            return self.qa_templates[question_lower]['exact_answer'], 1.0
        
        # 2. Поиск по ключевым словам
        best_match = None
        best_score = 0
        best_template_key = None
        
        for template_key, template_data in self.qa_templates.items():
            score = self.calculate_exact_match_score(question_lower, template_key)
            
            if score > best_score:
                best_score = score
                best_match = template_data['exact_answer']
                best_template_key = template_key
        
        # ВЫСОКИЙ ПОРОГ для точного копирования
        if best_match and best_score >= 0.7:  # 70% совпадение
            print(f"🎯 ВЫСОКОЕ СОВПАДЕНИЕ ({best_score:.1%}): '{best_template_key}'")
            return best_match, best_score
        
        # ДЕБАГ: Покажем топ-3 наиболее похожих вопросов
        print("🔍 Наиболее похожие вопросы:")
        scored_questions = []
        for template_key, template_data in self.qa_templates.items():
            score = self.calculate_exact_match_score(question_lower, template_key)
            if score > 0.3:
                scored_questions.append((score, template_key, template_data['exact_answer']))
        
        scored_questions.sort(reverse=True)
        for i, (score, tpl_key, answer) in enumerate(scored_questions[:3]):
            print(f"  {i+1}. '{tpl_key}' ({score:.1%})")
        
        return None, 0.0
    
    def calculate_exact_match_score(self, user_question, template_question):
        """Строгий расчет совпадения для точного копирования"""
        user_words = set(re.findall(r'\w+', user_question))
        template_words = set(re.findall(r'\w+', template_question))
        
        if not user_words or not template_words:
            return 0
        
        # Совпадение ключевых слов
        common_words = user_words.intersection(template_words)
        score = len(common_words) / len(template_words)
        
        return min(score, 1.0)
    
    def ask(self, question, log_interaction=True):
        """Основной метод вопрос-ответ с ПРИОРИТЕТОМ ТОЧНОГО КОПИРОВАНИЯ"""
        print(f"\n🔍 ВОПРОС: {question}")
        
        # 1. Пытаемся найти ТОЧНОЕ совпадение для копирования
        exact_answer, confidence = self.find_exact_match(question)
        
        if exact_answer:
            answer = exact_answer
            used_template = True
            answer_source = "exact_copy"
            print("✅ ИСПОЛЬЗУЮ ТОЧНЫЙ ОТВЕТ ИЗ БАЗЫ ЗНАНИЙ")
        else:
            # 2. Если точного совпадения нет - используем few-shot learning
            print("🔎 Точного совпадения нет, использую few-shot learning...")
            answer = self.generate_with_few_shot(question)
            used_template = False
            confidence = 0.5
            answer_source = "generated"
        
        # 3. Логируем взаимодействие
        if log_interaction:
            self.log_interaction(question, answer, used_template, confidence, answer_source=answer_source)
        
        return answer
    
    def generate_with_few_shot(self, question):
        """Генерирует ответ с помощью few-shot learning"""
        few_shot_context = "ПРИМЕРЫ ВОПРОСОВ И ОТВЕТОВ:\n\n"
        for i, example in enumerate(self.few_shot_examples[:3], 1):
            few_shot_context += f"Пример {i}:\n"
            few_shot_context += f"Вопрос: {example['question']}\n"
            few_shot_context += f"Ответ: {example['answer']}\n\n"
        
        strict_prompt = f"""
Ты - AI помощник службы поддержки. Твоя задача - давать точные ответы как в примерах ниже.

{few_shot_context}
НОВЫЙ ВОПРОС: {question}

ИНСТРУКЦИЯ СТРОГО:
1. ОТВЕЧАЙ ТОЧНО КАК В ПРИМЕРАХ ВЫШЕ
2. НЕ ПРИДУМЫВАЙ НИЧЕГО НОВОГО
3. БУДЬ МАКСИМАЛЬНО КОНКРЕТЕН
4. ЕСЛИ ИНФОРМАЦИИ НЕТ - СКАЖИ "Информация отсутствует в базе знаний"

ОТВЕТ:
"""
        
        try:
            response = ollama.generate(model="llama3.1:8b", prompt=strict_prompt)
            return response['response']
        except Exception as e:
            return f"❌ Ошибка: {e}"

# Тестирование
if __name__ == "__main__":
    rag = TrainedRAGWithExactCopy(
        knowledge_file="/home/sl1m/hacatons/mosprom/SAMARKAND_HACATON/rosatom_ml/scripts/Обращения.txt",
        answers_dir="/home/sl1m/hacatons/mosprom/SAMARKAND_HACATON/rosatom_ml/scripts/knowledge_qa_files",
        categories_index_file="/home/sl1m/hacatons/mosprom/SAMARKAND_HACATON/rosatom_ml/scripts/categories_index.json",
        logs_dir="/home/sl1m/hacatons/mosprom/SAMARKAND_HACATON/rosatom_ml/scripts/logs"
    )
    
    # Тестируем конкретный вопрос
    test_question = "Где найти статус поставки?"
    print(f"\n🧪 ТЕСТИРУЕМ ВОПРОС: '{test_question}'")
    answer = rag.ask(test_question)
    print(f"💡 ОТВЕТ: {answer}")