import ollama

# Самый простой классификатор
def simple_classify(text):
    prompt = f"""
    К какой категории отнести: "{text}"?
    Варианты: it, software, access, hr, finance, hardware, documents, devops, office, urgent
    
    Ответь одним словом.
    """
    
    response = ollama.generate(model="llama3.1:8b", prompt=prompt)
    return response['response'].strip()

# Читаем и обрабатываем файл
with open('/home/sl1m/hacatons/mosprom/SAMARKAND_HACATON/rosatom_ml/scripts/Обращения.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line and len(line) > 10:
            category = simple_classify(line)
            print(f"{category}: {line[:50]}...")