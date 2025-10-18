import ollama
import re
import json
from pathlib import Path

class ModelTrainer:
    def __init__(self, answers_dir, categories_index_file):
        self.answers_dir = Path(answers_dir)
        self.categories_index = self.load_categories_index(categories_index_file)
    
    def load_categories_index(self, categories_index_file):
        """Загружает индекс категорий"""
        try:
            with open(categories_index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Ошибка загрузки категорий: {e}")
            return {"domains": {}}
    
    def prepare_training_data(self):
        """Подготавливает данные для обучения"""
        training_data = []
        
        for domain_name, domain_data in self.categories_index['domains'].items():
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
                        training_data.extend(self.parse_training_examples(file_path))
        
        return training_data
    
    def parse_training_examples(self, file_path):
        """Парсит файлы для создания примеров обучения"""
        examples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Разделяем на вопросы-ответы
            sections = content.split('\n\n')
            
            for section in sections:
                if section.strip():
                    lines = section.strip().split('\n')
                    if len(lines) >= 2:
                        question = lines[0].strip().rstrip('?:')
                        answer = '\n'.join(lines[1:]).strip()
                        
                        if question and answer:
                            # Создаем пример для обучения
                            training_example = {
                                "input": question,
                                "output": answer
                            }
                            examples.append(training_example)
            
            print(f"📚 Подготовлено {len(examples)} примеров из {file_path.name}")
            
        except Exception as e:
            print(f"❌ Ошибка парсинга {file_path}: {e}")
        
        return examples
    
    def create_modelfile(self, training_data, model_name="support-assistant"):
        """Создает правильный Modelfile"""
        modelfile_content = f"""FROM llama3.2:1b

SYSTEM \"\"\"
Ты - AI помощник службы поддержки компании. Твоя задача - давать точные, конкретные ответы на вопросы сотрудников.

ИНСТРУКЦИИ:
1. Отвечай ТОЛЬКО на основе предоставленной информации
2. Будь максимально конкретным и точным
3. Не добавляй лишних объяснений
4. Если не знаешь ответ - скажи "Информация отсутствует в базе знаний"
5. Используй только проверенные данные

ФОРМАТ ОТВЕТА:
[Конкретный ответ]
[При необходимости - шаги решения]
\"\"\"
"""
        
        # Добавляем примеры в формате для обучения
        for example in training_data[:50]:  # Ограничиваем для скорости
            modelfile_content += f"""
# Вопрос: {example['input']}
# Ответ: {example['output']}
"""
        
        # Сохраняем Modelfile
        with open("Modelfile", "w", encoding="utf-8") as f:
            f.write(modelfile_content)
        
        print("✅ Modelfile создан")
        return "Modelfile"
    
    def train_model(self):
        """Запускает процесс обучения"""
        print("🔄 Подготовка данных для обучения...")
        training_data = self.prepare_training_data()
        
        if not training_data:
            print("❌ Нет данных для обучения")
            return
        
        print(f"📊 Всего примеров для обучения: {len(training_data)}")
        
        # Создаем Modelfile
        modelfile_path = self.create_modelfile(training_data)
        
        # Обучаем модель ПРАВИЛЬНЫМ способом
        print("🎯 Начинаем обучение модели...")
        try:
            # Способ 1: Через создание модели из Modelfile
            response = ollama.create(
                model="support-assistant",
                modelfile=open(modelfile_path, 'r', encoding='utf-8').read()
            )
            print("✅ Модель успешно обучена!")
            print(f"📝 Имя модели: support-assistant")
            return "support-assistant"
        except Exception as e:
            print(f"❌ Ошибка обучения: {e}")
            print("🔄 Пробуем альтернативный способ...")
            return self.train_model_alternative(training_data)
    
    def train_model_alternative(self, training_data):
        """Альтернативный способ обучения через few-shot prompting"""
        print("🔄 Использую альтернативный подход...")
        
        # Создаем улучшенную версию RAG с обученными промптами
        trained_prompts = {}
        
        for example in training_data:
            trained_prompts[example['input']] = example['output']
        
        # Сохраняем обученные промпты
        with open("trained_prompts.json", "w", encoding="utf-8") as f:
            json.dump(trained_prompts, f, ensure_ascii=False, indent=2)
        
        print("✅ Обученные промпты сохранены в trained_prompts.json")
        return "trained_prompts"

# Запуск обучения
if __name__ == "__main__":
    trainer = ModelTrainer(
        answers_dir="/home/sl1m/hacatons/mosprom/SAMARKAND_HACATON/rosatom_ml/scripts/knowledge_qa_files",
        categories_index_file="/home/sl1m/hacatons/mosprom/SAMARKAND_HACATON/rosatom_ml/scripts/categories_index.json"
    )
    
    trained_model = trainer.train_model()