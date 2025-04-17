# ComfyUI‑VLMStudio 🔮  
*A multimodal Gemma node for ComfyUI (extension for **ComfyUI‑EBU‑LMStudio**)*

<!-- Badges -->
[![GitHub stars](https://img.shields.io/github/stars/olyyarm/ComfyUI-VLMStudio?style=social&logo=github)](https://github.com/olyyarm/ComfyUI-VLMStudio)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?logo=opensourceinitiative)](LICENSE)

---

## ✨ Что это такое?
`ComfyUI‑VLMStudio` — кастом‑нода **Gemma Multimodal Analyzer** для ComfyUI.  
Она добавляет *визуально‑текстовый* анализ изображений через LM Studio (локальный эндпоинт) и модель **Gemma‑3‑12b‑it**.

> **TL;DR**: ставишь базовый пакет [ComfyUI‑EBU‑LMStudio](https://github.com/burnsbert/ComfyUI-EBU-LMStudio) —  
> клонируешь этот репо — получаешь ноду, которая отправляет картинку + текст в LM Studio и возвращает ответ LLM.

---

## 🔗 Почему отдельный репозиторий?
Пакет **burnsbert/ComfyUI‑EBU‑LMStudio** даёт текстовый чат с LM Studio.  
`ComfyUI‑VLMStudio` дополняет его мультимодальностью:

|                        | EBU‑LMStudio | **VLMStudio** |
|------------------------|--------------|---------------|
| Текст ↔ LLM            | ✅           | ✅           |
| Картинка + текст ↔ LLM | ❌           | ✅           |
| Авто‑base64            | ❌           | ✅           |
| Валидация изображений  | ❌           | ✅           |

---

## ⚡ Быстрый старт

```bash
# 1. Базовый текстовый пакет
git clone https://github.com/burnsbert/ComfyUI-EBU-LMStudio.git

# 2. Мультимодальный аддон
cd ComfyUI/custom_nodes
git clone https://github.com/olyyarm/ComfyUI-VLMStudio.git
