import subprocess
import sys
import re
import comfy.model_management as model_management
import gc
import torch
import requests
import time
import random

class GemmaMultimodalAnalyzer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", { "multiline": True, "default": "" }),
                "system_message": ("STRING", { "multiline": True, "default": 
                    "You are a visual assistant. Analyze the image and answer questions about it."
                }),
                "url": ("STRING", { "multiline": False, "default": "http://127.0.0.1:1234/v1/chat/completions" }),
                "max_tokens": ("INT", {"default": 300, "min": 10, "max": 100000}),
                "temp": ("FLOAT", {"default": 0.7, "min": 0, "max": 1, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0, "max": 1, "step": 0.01}),
            },
            # А здесь делаем приём картинки опциональным. Если нужно — сделайте required
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "generateText"
    CATEGORY = "LMStudio"

    def generateText(self, 
                     prompt, 
                     system_message, 
                     url, 
                     max_tokens, 
                     temp, 
                     top_p,
                     image=None):
        """
        Основная функция: если картинка подана, конвертируем в base64
        и отправляем её вместе с prompt. Если картинки нет, шлём только текст.
        """
        if image is not None:
            # Проверим формат (убедимся, что это не латент на 768 каналов)
            self.validate_image(image)
            # Конвертируем в base64 для передачи LLM
            image_data = self.tensor_to_base64(image)
        else:
            image_data = None

        # Сформируем запрос к LLM (Gemma-3-12b-it)
        description = self.call_api(
            prompt_text = prompt,
            system_message = system_message,
            url = url,
            max_tokens = max_tokens,
            temp = temp,
            top_p = top_p,
            image_data = image_data
        )

        return (description,)

    def validate_image(self, tensor):
        """
        Проверяет, что это 4D тензор с 1, 3 или 4 каналами,
        а не латент с 768 каналами.
        """
        if tensor.dim() != 4:
            raise ValueError("Invalid image format: expected a 4D tensor (B, C, H, W) or (B, H, W, C).")

        c1 = tensor.shape[1]
        c2 = tensor.shape[3]
        # Проверяем, не латент ли (768 каналов)
        if c1 in [1,3,4] or c2 in [1,3,4]:
            return
        else:
            raise ValueError(
                f"Invalid number of channels in image: {tensor.shape}. "
                "Expected 1, 3, or 4 channels. Perhaps you fed a latent (768 channels)?"
            )

    def tensor_to_base64(self, tensor):
        """
        Переводит ComfyUI-формат [B, C, H, W] или [B, H, W, C] 
        в base64-encoded PNG.
        """
        import torch
        from torchvision.transforms import ToPILImage
        import io, base64

        # Если [B, H, W, C], меняем на [B, C, H, W]
        if tensor.shape[1] not in [1,3,4] and tensor.shape[3] in [1,3,4]:
            tensor = tensor.permute(0, 3, 1, 2)
        
        # Сжимаем [0..1] float -> [0..255] byte
        tensor = torch.clamp(tensor * 255.0, 0, 255).byte()

        pil_img = ToPILImage()(tensor[0].cpu())  # Берём первую картинку из батча
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def build_payload(self, prompt_text, system_message, image_data, temp, max_tokens, top_p):
        """
        Собираем JSON, который понимает ваша LLM gemma-3-12b-it
        (убедитесь, что формат messages совпадает с тем, 
        как LM Studio/Gemma реально принимает изображение + текст).
        """
        user_content = []
        user_content.append({"type": "text", "text": prompt_text})

        if image_data:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data}"
                }
            })

        payload = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user",   "content": user_content}
            ],
            "temperature": temp,
            "max_tokens": max_tokens,
            "top_p": top_p
        }
        return payload

    def call_api(self, prompt_text, system_message, url, max_tokens, temp, top_p, image_data):
        """
        Отправляем POST-запрос на LMStudio-endpoint с мультимодальными данными
        и возвращаем текст-ответ.
        """
        import json

        payload = self.build_payload(
            prompt_text, 
            system_message, 
            image_data,
            temp,
            max_tokens,
            top_p
        )

        print(f"API Call: {url}")
        print("Request Payload:", json.dumps(payload, indent=2))

        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                result_json = response.json()
                # Предполагаем, что LLM Studio возвращает это поле
                return result_json["choices"][0]["message"]["content"]
            else:
                err = f"API error {response.status_code}: {response.text}"
                print(err)
                return err
        except Exception as e:
            err = f"Request error: {str(e)}"
            print(err)
            return err

# Делаем доступным наш класс в ComfyUI
NODE_CLASS_MAPPINGS = {
    "GemmaMultimodalAnalyzer": GemmaMultimodalAnalyzer
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GemmaMultimodalAnalyzer": "🔮 Gemma Multimodal Analyzer"
}
