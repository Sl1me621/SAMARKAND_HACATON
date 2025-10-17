# model/llama_classifier.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Dict, List
import re

class LlamaClassifier:
    def __init__(self):
        self.model_name = "meta-llama/Llama-3.1-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.categories = [
            "it_systems", "accounting", "hardware", 
            "hr", "access", "integration", "other"
        ]
        
        # Системный промпт для классификации
        self.system_prompt = """Ты - AI ассистент для классификации обращений в техподдержку Росатома. 
        Классифицируй обращение в одну из категорий:
        - it_systems: Jenkins, Docker, GitLab, VPN, сети, серверы, CI/CD
        - accounting: 1С, НДС, проводки, отчетность, бухгалтерия, налоги
        - hardware: принтеры, ноутбуки, мышки, мониторы, оборудование
        - hr: отпуска, больничные, справки, графики работы, кадры
        - access: доступы, права, пароли, учетные записи, AD
        - integration: интеграции между системами, CRM, API, синхронизация
        - other: всё остальное
        
        Ответь ТОЛЬКО названием категории без пояснений."""
    
    def classify_text(self, text: str) -> Dict:
        """Классификация текста с помощью Llama"""
        prompt = f"{self.system_prompt}\n\nОбращение: {text}\nКатегория:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        category = self._extract_category(response)
        
        return {
            "category": category,
            "confidence": 0.9,  # Для простоты, можно добавить вычисление уверенности
            "full_response": response
        }
    
    def _extract_category(self, response: str) -> str:
        """Извлекает категорию из ответа модели"""
        # Ищем категорию в ответе
        for category in self.categories:
            if category in response.lower():
                return category
        
        # Если не нашли, используем эвристики
        return "other"