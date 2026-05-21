import torch
import clip
from PIL import Image
from typing import Union


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def encode_images(images: list[Union[Image.Image, str]]) -> torch.Tensor:
    """
    Кодирует список изображений в CLIP-эмбеддинги.
    
    Args:
        images (list): список объектов PIL.Image.Image или путей к изображениям.
        
    Returns:
        torch.Tensor: матрица эмбеддингов [len(images), d]
    """
    processed = []
    for img in images:
        if isinstance(img, str):  # если путь к файлу
            img = Image.open(img)#.convert("RGB")
        processed.append(preprocess(img))

    image_input = torch.stack(processed).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features


def encode_text(texts: list[str]) -> torch.Tensor:
    """
    Кодирует список текстов в CLIP-эмбеддинги.
    
    Args:
        texts (list): список строк
        
    Returns:
        torch.Tensor: матрица эмбеддингов [len(texts), d]
    """
    
    text_tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features


def reconstruction_score(generated_images: list[Union[Image.Image, str]], train_images: list[Union[Image.Image, str]]) -> float:
    """
    Считает Reconstruction Score:
    среднее попарное CLIP-сходство между сгенерированными и обучающими изображениями.
    
    Args:
        generated_images (list): список сгенерированных PIL.Image.Image или путей к файлам
        train_images (list): список реальных PIL.Image.Image или путей к файлам
        
    Returns:
        score (float): Reconstruction Score
    """
    
    gen_embeds = encode_images(generated_images)
    real_embeds = encode_images(train_images)

    sims = gen_embeds @ real_embeds.T 
    score = sims.mean().item()
    return score


def editability_score(generated_images: list[Union[Image.Image, str]], prompt_text: str) -> float:
    """
    Считает Editability Score:
    косинусное сходство между средним эмбеддингом изображений и эмбеддингом текста.
    
    Args:
        generated_images (list): список сгенерированных PIL.Image.Image или путей к файлам
        prompt_text (str): текстовый промпт БЕЗ placeholder
        
    Returns:
        score (float): Editability Score
    """
    
    gen_embeds = encode_images(generated_images)
    avg_img_embed = gen_embeds.mean(dim=0, keepdim=True)
    avg_img_embed /= avg_img_embed.norm(dim=-1, keepdim=True)

    text_embed = encode_text([prompt_text])

    score = (avg_img_embed @ text_embed.T).item()
    return score