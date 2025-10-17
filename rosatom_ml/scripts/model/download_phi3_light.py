# scripts/download_qwen_wget.py
import os
import subprocess
import time

def download_qwen_wget():
    """Скачиваем Qwen2-1.5B через wget"""
    model_name = "Qwen2-1.5B-Instruct"
    local_dir = "models/qwen2-1.5b"
    
    # Создаем папку
    os.makedirs(local_dir, exist_ok=True)
    
    print("📥 Скачиваем Qwen2-1.5B-Instruct через wget...")
    print("💾 Размер: ~3GB")
    print("⏳ Это займет некоторое время...")
    
    # Файлы для скачивания
    files = [
        "config.json",
        "generation_config.json", 
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json"
    ]
    
    base_url = "https://huggingface.co/Qwen/Qwen2-1.5B-Instruct/resolve/main"
    
    for file in files:
        file_path = f"{local_dir}/{file}"
        file_url = f"{base_url}/{file}"
        
        print(f"\n⬇️  Скачиваем {file}...")
        
        try:
            # Используем wget с ресайзом и прогресс-баром
            result = subprocess.run([
                "wget", 
                "-c",  # Продолжить скачивание
                "-O", file_path,
                file_url
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {file} скачан!")
            else:
                print(f"⚠️  Проблема с {file}: {result.stderr}")
                # Пробуем альтернативный URL
                print("🔄 Пробуем альтернативный URL...")
                alt_result = subprocess.run([
                    "wget",
                    "-O", file_path, 
                    f"https://huggingface.co/Qwen/Qwen2-1.5B-Instruct/resolve/main/{file}?download=true"
                ], capture_output=True, text=True)
                
                if alt_result.returncode == 0:
                    print(f"✅ {file} скачан через альтернативный URL!")
                else:
                    print(f"❌ Не удалось скачать {file}")
                    
        except Exception as e:
            print(f"❌ Ошибка: {e}")
    
    print(f"\n🎉 Проверяем скачанные файлы...")
    
    # Проверяем что файлы существуют
    downloaded_files = os.listdir(local_dir)
    print(f"📁 Скачано файлов: {len(downloaded_files)}")
    for file in downloaded_files:
        size = os.path.getsize(f"{local_dir}/{file}") / (1024*1024)
        print(f"   {file}: {size:.1f}MB")
    
    return len(downloaded_files) > 3  # Хотя бы основные файлы

def download_smaller_files():
    """Скачиваем только самые важные файлы"""
    local_dir = "models/qwen2-1.5b-minimal"
    os.makedirs(local_dir, exist_ok=True)
    
    print("\n🔄 Скачиваем минимальную версию...")
    
    essential_files = {
        "config.json": "https://huggingface.co/Qwen/Qwen2-1.5B-Instruct/resolve/main/config.json",
        "tokenizer.json": "https://huggingface.co/Qwen/Qwen2-1.5B-Instruct/resolve/main/tokenizer.json",
        "tokenizer_config.json": "https://huggingface.co/Qwen/Qwen2-1.5B-Instruct/resolve/main/tokenizer_config.json",
    }
    
    for file_name, file_url in essential_files.items():
        file_path = f"{local_dir}/{file_name}"
        
        print(f"⬇️  {file_name}...")
        result = subprocess.run([
            "wget", "-O", file_path, file_url
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {file_name} скачан!")
        else:
            print(f"❌ Ошибка: {result.stderr}")
    
    return local_dir

if __name__ == "__main__":
    # Пробуем скачать полную версию
    success = download_qwen_wget()
    
    if not success:
        print("\n🔄 Полная версия не скачалась, пробуем минимальную...")
        minimal_dir = download_smaller_files()
        print(f"📁 Минимальная версия в: {minimal_dir}")