# P-LoRA
В этом репозитории предоставлен код моих экспериментов

`ImageIdit` - P-LoRA эксперименты, обучения, подсчет метрик.  
`T-LoRA` - подсчет метрик. Для обучения использовался код из [оригинального репозитория]([url](https://github.com/ControlGenAI/T-LoRA))  
`data_train` - данные для обучения во few shot режиме  
`data_train_one` - данные для обучения в one shot режиме

---
# ImageEdit

В каждой папке этой группы лежит свой `train_....py` и ноутбук `dynamic_metrics.ipynb` (или `metricks.ipynb`) для расчёта editability/reconstruction score по чекпоинтам. После запуска в той же папке появляются `checkpoint-100/`…`checkpoint-1000/` (LoRA-веса каждые 100 шагов) и `sample/` (сгенерированные изображения для оценки).

| Папка | Конфигурация | Режим |
|---|---|---|
| `base_dynamic/` | DreamBooth| few-shot |
| `base_dynamic_one_shot/` | DreamBooth | one-shot |
| `mult_1_dynamic/` | Left P-LoRA | few-shot |
| `mult_1_dynamic_one_shot/` | Left P-LoRA | one-shot |
| `mult_1_right_dynamic/` | Right P-LoRA | few-shot |
| `mult_1_right_dynamic_one_shot/` | Right P-LoRA | one-shot |
| `mix_dynamic/` | **Mixture P-LoRA** — общий скрипт + `train_all.sh` + `metricks.ipynb` | few-shot |
| `mix_025_dynamic/` | Mixture, λ=0.25 | few-shot |
| `mix_050_dynamic/` | Mixture, λ=0.50 | few-shot |
| `mix_075_dynamic/` | Mixture, λ=0.75 | few-shot |
| `beta_dynamics/` | **U-Shape + Center P-LoRA** — общий скрипт, `train_all.sh`, `train_one_shot.sh`, `metricks.ipynb` | оба режима |
| `beta_dynamics/u_shaped_05_05/` | U-Shape | few-shot |
| `beta_dynamics/u_shaped_05_05_one_shot/` | U-Shape | one-shot |
| `beta_dynamics/center_5_5/` | Center | few-shot |
| `beta_dynamics/center_5_5_one_shot/` | Center | one-shot |
| `base_dreambooth/` | DreamBooth | — |

### Общий код и утилиты

| Путь | Содержимое |
|---|---|
| `src/metrics.py` |  `reconstruction_score`, `editability_score` |
| `utils/visual.py` | `make_grid_with_caption` и `make_training_progress_grid` для построения сеток изображений |
| `data/test_prompts.py` | `live_set`, `object_set`, `base_prompt` |
