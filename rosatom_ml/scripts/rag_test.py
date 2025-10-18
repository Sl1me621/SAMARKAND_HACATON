import ollama
import re
import os
from pathlib import Path
from collections import Counter

class EnhancedRAG:
    def __init__(self, knowledge_file, answers_dir):
        self.knowledge_file = knowledge_file
        self.answers_dir = Path(answers_dir)
        self.documents = self.load_documents()
        self.qa_database = self.load_qa_database()
        print(f"Загружено {len(self.documents)} обращений и {len(self.qa_database)} QA пар")
    
    def load_documents(self):
        """Загружает обращения из файла"""
        try:
            with open(self.knowledge_file, 'r', encoding='utf-8') as f:
                documents = [line.strip() for line in f if line.strip() and len(line.strip()) > 20]
            return documents
        except Exception as e:
            print(f"Ошибка загрузки файла обращений: {e}")
            return []
    
    def load_qa_database(self):
        """Загружает базу вопросов-ответов из структуры папок"""
        qa_database = {}
        
        if not self.answers_dir.exists():
            print(f"Папка с ответами не найдена: {self.answers_dir}")
            return qa_database
        
        # Проходим по всем папкам и подпапкам
        for topic_dir in self.answers_dir.iterdir():
            if topic_dir.is_dir():
                topic_name = topic_dir.name
                
                # Ищем файлы .md и .txt в подпапках
                for file_path in topic_dir.rglob("*.md"):
                    qa_database.update(self.parse_qa_file(file_path, topic_name))
                
                for file_path in topic_dir.rglob("*.txt"):
                    qa_database.update(self.parse_qa_file(file_path, topic_name))
        
        return qa_database
    
    def parse_qa_file(self, file_path, topic):
        """Парсит файл с вопросами-ответами"""
        qa_pairs = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Разные стратегии парсинга в зависимости от формата
            if file_path.suffix == '.md':
                # Для Markdown файлов - ищем заголовки и контент под ними
                sections = re.split(r'#+\s+', content)
                for section in sections[1:]:  # пропускаем первую пустую секцию
                    if section.strip():
                        lines = section.strip().split('\n')
                        question = lines[0].strip()
                        answer = '\n'.join(lines[1:]).strip()
                        if question and answer:
                            qa_pairs[question] = {
                                'answer': answer,
                                'topic': topic,
                                'subtopic': file_path.parent.name,
                                'source': file_path.name
                            }
            else:
                # Для TXT файлов - простой формат вопрос: ответ
                lines = content.split('\n')
                current_question = None
                current_answer = []
                
                for line in lines:
                    line = line.strip()
                    if line.endswith('?') or line.endswith(':'):
                        # Сохраняем предыдущую пару
                        if current_question and current_answer:
                            qa_pairs[current_question] = {
                                'answer': '\n'.join(current_answer),
                                'topic': topic,
                                'subtopic': file_path.parent.name,
                                'source': file_path.name
                            }
                        
                        current_question = line.rstrip('?:')
                        current_answer = []
                    elif current_question and line:
                        current_answer.append(line)
                
                # Добавляем последнюю пару
                if current_question and current_answer:
                    qa_pairs[current_question] = {
                        'answer': '\n'.join(current_answer),
                        'topic': topic,
                        'subtopic': file_path.parent.name,
                        'source': file_path.name
                    }
            
            print(f"Загружено {len(qa_pairs)} QA пар из {file_path}")
            
        except Exception as e:
            print(f"Ошибка чтения файла {file_path}: {e}")
        
        return qa_pairs
    
    def find_relevant_qa(self, query, top_k=3):
        """Находит релевантные вопросы-ответы"""
        query_words = set(re.findall(r'\w+', query.lower()))
        
        scored_qa = []
        for question, qa_data in self.qa_database.items():
            question_words = set(re.findall(r'\w+', question.lower()))
            common_words = query_words.intersection(question_words)
            score = len(common_words)
            
            # Бонус за совпадение в ответе
            answer_words = set(re.findall(r'\w+', qa_data['answer'].lower()))
            score += len(query_words.intersection(answer_words)) * 0.5
            
            if score > 0:
                scored_qa.append((score, question, qa_data))
        
        scored_qa.sort(reverse=True)
        return [(q, data) for score, q, data in scored_qa[:top_k]]
    
    def find_relevant_documents(self, query, top_k=2):
        """Находит релевантные обращения"""
        query_words = set(re.findall(r'\w+', query.lower()))
        
        scored_docs = []
        for doc in self.documents:
            doc_words = set(re.findall(r'\w+', doc.lower()))
            common_words = query_words.intersection(doc_words)
            score = len(common_words)
            if score > 0:
                scored_docs.append((score, doc))
        
        scored_docs.sort(reverse=True)
        return [doc for score, doc in scored_docs[:top_k]]
    
    def ask(self, question):
        """Задает вопрос системе RAG"""
        if not self.qa_database and not self.documents:
            return "База знаний пуста"
        
        # Ищем релевантные QA пары
        relevant_qa = self.find_relevant_qa(question)
        relevant_docs = self.find_relevant_documents(question)
        
        if not relevant_qa and not relevant_docs:
            return "Не найдено релевантной информации в базе знаний"
        
        # Формируем контекст
        context_parts = []
        
        if relevant_qa:
            context_parts.append("=== БАЗА ЗНАНИЙ (вопросы-ответы) ===")
            for i, (q, qa_data) in enumerate(relevant_qa, 1):
                context_parts.append(f"{i}. ВОПРОС: {q}")
                context_parts.append(f"   ОТВЕТ: {qa_data['answer']}")
                context_parts.append(f"   Тема: {qa_data['topic']} -> {qa_data['subtopic']}")
                context_parts.append("")
        
        if relevant_docs:
            context_parts.append("=== ПОХОЖИЕ ОБРАЩЕНИЯ ===")
            for i, doc in enumerate(relevant_docs, 1):
                context_parts.append(f"{i}. {doc}")
        
        context = "\n".join(context_parts)
        
        prompt = f"""
        Ты - AI помощник службы поддержки. Ты должен использовать данные из базы, найде
        
        {context}
        
        НОВЫЙ ВОПРОС ПОЛЬЗОВАТЕЛЯ: {question}
        
        ИНСТРУКЦИЯ:
        1. Используй информацию из базы знаний для точного ответа
        2. Если есть точный ответ в базе - используй его
        3. Если информации недостаточно - предложи общие рекомендации
        4. Будь конкретен и полезен
        5. Указывай конкретные шаги решения
        
        ОТВЕТ:
        """
        
        try:
            response = ollama.generate(model="llama3.1:8b", prompt=prompt)
            return response['response']
        except Exception as e:
            return f"Ошибка генерации ответа: {e}"
    
    def get_statistics(self):
        """Возвращает статистику по загруженным данным"""
        topics = Counter()
        subtopics = Counter()
        
        for qa_data in self.qa_database.values():
            topics[qa_data['topic']] += 1
            subtopics[qa_data['subtopic']] += 1
        
        return {
            'total_questions': len(self.qa_database),
            'total_documents': len(self.documents),
            'topics': dict(topics),
            'subtopics': dict(subtopics)
        }

def main():
    # Настройки путей
    KNOWLEDGE_FILE = "/home/sl1m/hacatons/mosprom/SAMARKAND_HACATON/rosatom_ml/scripts/Обращения.txt"
    ANSWERS_DIR = "/home/sl1m/hacatons/mosprom/SAMARKAND_HACATON/rosatom_ml/scripts/knowledge_qa_files"
    # Создаем систему RAG
    print("🔄 ЗАГРУЗКА RAG СИСТЕМЫ...")
    rag = EnhancedRAG(KNOWLEDGE_FILE, ANSWERS_DIR)
    
    # Показываем статистику
    stats = rag.get_statistics()
    print(f"\n📊 СТАТИСТИКА:")
    print(f"   Вопросов-ответов: {stats['total_questions']}")
    print(f"   Обращений: {stats['total_documents']}")
    print(f"   Тем: {len(stats['topics'])}")
    print(f"   Подтем: {len(stats['subtopics'])}")
    
    # Тестовые вопросы из файла обращений
    print(f"\n🔍 ВЫБИРАЕМ ТЕСТОВЫЕ ВОПРОСЫ ИЗ ФАЙЛА...")
    test_questions = rag.documents[:10]  # Берем первые 10 обращений для теста
    
    print("=== ТЕСТИРОВАНИЕ RAG СИСТЕМЫ ===\n")
    
    for i, question in enumerate(test_questions, 1):
        print(f"🧪 ТЕСТ {i}/{len(test_questions)}")
        print(f"❓ ВОПРОС: {question}")
        
        answer = rag.ask(question)
        print(f"💡 ОТВЕТ: {answer}")
        print("-" * 100)

def interactive_mode():
    """Интерактивный режим вопрос-ответ"""
    KNOWLEDGE_FILE = "/home/sl1m/hacatons/mosprom/SAMARKAND_HACATON/rosatom_ml/scripts/Обращения.txt"
    ANSWERS_DIR = "/home/sl1m/hacatons/mosprom/SAMARKAND_HACATON/rosatom_ml/scripts/knowledge_qa_files"
    rag = EnhancedRAG(KNOWLEDGE_FILE, ANSWERS_DIR)
    
    print("🤖 RAG СИСТЕМА ГОТОВА К РАБОТЕ")
    print("Задавайте вопросы (для выхода введите 'выход')\n")
    
    while True:
        question = input("👤 ВАШ ВОПРОС: ").strip()
        
        if question.lower() in ['выход', 'exit', 'quit']:
            print("До свидания!")
            break
        
        if not question:
            continue
            
        print("🔍 Поиск в базе знаний...")
        answer = rag.ask(question)
        print(f"🤖 ОТВЕТ: {answer}\n")

if __name__ == "__main__":
    # Запуск тестового режима
    main()
    
    # Раскомментируйте для интерактивного режима
    # interactive_mode()