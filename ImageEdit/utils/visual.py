from PIL import Image, ImageDraw, ImageFont
import os


def make_grid_with_caption(images: list[Image.Image], caption: str, grid_size=(1, 5), padding=20, bg_color=(255, 255, 255)) -> Image.Image:
    """ 
    Отрисовываем сетку мзображений с подписью
    
    Args:
        images (list[Image.Image]): список изображений
        caption: (str): подпись
        grid_size (tuple[int, int]): расположение на сетке
        padding (int): обрамление
        bg_color (tuple): цвет фона
    
    Return:
        grid_img (list[Image.Image]): изображение сетки
    """
    rows, cols = grid_size
    img_w, img_h = images[0].size
    
    # Шрифт
    font = ImageFont.truetype("DejaVuSans.ttf", 100)
    
    # Определяем размер текста
    dummy_img = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), caption, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    # Размер итогового полотна
    grid_width = cols * img_w + (cols + 1) * padding
    grid_height = rows * img_h + (rows + 1) * padding + text_h + padding
    
    grid_img = Image.new("RGB", (grid_width, grid_height), bg_color)
    draw = ImageDraw.Draw(grid_img)
    
    # Рисуем картинки
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        x = padding + c * (img_w + padding)
        y = padding + r * (img_h + padding)
        grid_img.paste(img, (x, y))
    
    # Подпись снизу
    text_x = (grid_width - text_w) // 2
    text_y = grid_height - text_h - padding
    draw.text((text_x, text_y), caption, font=font, fill=(0, 0, 0))
    
    return grid_img

def make_training_progress_grid(
    steps: list[int],
    prompt_data: dict,
    base_path: str = ".",
    grid_size: tuple = (1, 5),
    padding: int = 80,
    bg_color: tuple = (255, 255, 255),
    title_font_size: int = 90,
    label_font_size: int = 60,
    max_title_chars: int = 100,
) -> Image.Image:
    """
    Args:
        steps (list[int]): список шагов для отрисовки
        prompt_data (dics): словарь с промптом ({'prompt': 'a purple sks dog', 'idx': 1, 'set': 'live'})
        base_path (str): папка для поиска
        grid_size (tuple[int, int]): размер сетки 
        padding (int): паддинг между элементами сетки
        bg_color (tuple[int, int, int]): цвет фона (255, 255, 255) 
        title_font_size (int): размер названия
        label_font_size (int): размер подписи
        max_title_chars (int): максимлаьнок кол-во символов для подписи
    
    Returns:
        grid (PIL.Image.Image): финальное изображение
    """
    rows, cols = grid_size

    images = []
    for step in steps:
        img_path = os.path.join(
            base_path, f"checkpoint-{step}", "sample",
            f"{prompt_data['set']}_{prompt_data['idx']}.png"
        )
        if os.path.exists(img_path):
            images.append(Image.open(img_path))
        else:
            if images:
                images.append(Image.new("RGB", images[0].size, (200, 200, 200)))
            else:
                images.append(Image.new("RGB", (512, 512), (200, 200, 200)))

    if not images:
        return Image.new("RGB", (512, 512), bg_color)

    img_w, img_h = images[0].size

    title_font = ImageFont.truetype("DejaVuSans.ttf", title_font_size)
    label_font = ImageFont.truetype("DejaVuSans.ttf", label_font_size)

    title = prompt_data["prompt"]
    if len(title) > max_title_chars:
        title = title[:max_title_chars] + "..."

    _tmp = Image.new("RGB", (10, 10))
    _d = ImageDraw.Draw(_tmp)

    title_w, title_h = _d.textbbox((0, 0), title, font=title_font)[2:]
    step_sample = f"Step: {max(steps) if steps else 0}"
    step_w, step_h = _d.textbbox((0, 0), step_sample, font=label_font)[2:]

    top_offset = padding + title_h + padding
    bottom_extra = (10 + step_h + padding) if rows > 0 else padding

    grid_width = cols * img_w + (cols + 1) * padding
    grid_height = top_offset + rows * img_h + (rows + 1) * padding + bottom_extra - padding

    grid_img = Image.new("RGB", (grid_width, grid_height), bg_color)
    draw = ImageDraw.Draw(grid_img)

    text_x = (grid_width - title_w) // 2
    text_y = padding
    draw.text((text_x, text_y), title, font=title_font, fill=(0, 0, 0))

    for idx, (img, step) in enumerate(zip(images, steps)):
        r, c = divmod(idx, cols)
        x = padding + c * (img_w + padding)
        y = top_offset + r * (img_h + padding)

        grid_img.paste(img, (x, y))
        step_text = f"Step: {step}"
        st_w = draw.textbbox((0, 0), step_text, font=label_font)[2]
        step_x = x + (img_w - st_w) // 2
        step_y = y + img_h + 10
        draw.text((step_x, step_y), step_text, font=label_font, fill=(0, 0, 0))

    return grid_img