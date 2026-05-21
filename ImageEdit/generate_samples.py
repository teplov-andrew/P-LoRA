import argparse
import sys
from datetime import timedelta
from pathlib import Path

import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from diffusers import StableDiffusionXLPipeline
from tqdm.auto import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", type=str, default=".",
                   help="Dir containing checkpoint-* subdirs (default: cwd).")
    p.add_argument("--steps", type=int, nargs="+",
                   default=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    p.add_argument("--n_gen", type=int, default=15,
                   help="Samples per prompt.")
    p.add_argument("--model_id", type=str,
                   default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--prompts_module", type=str, default="data.test_prompts",
                   help="Python module exposing live_set, object_set.")
    return p.parse_args()


def main():
    args = parse_args()
    pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=2))
    accelerator = Accelerator(kwargs_handlers=[pg_kwargs])
    rank = accelerator.process_index
    world = accelerator.num_processes
    device = accelerator.device

    exp_dir = Path(args.exp_dir).resolve()
    project_root = exp_dir.parent
    sys.path.insert(0, str(project_root))
    from importlib import import_module
    test_prompts = import_module(args.prompts_module)
    live_set = test_prompts.live_set
    object_set = test_prompts.object_set

    prompts = (
        [{"prompt": p, "idx": i, "set": "live"} for i, p in enumerate(live_set)]
        + [{"prompt": p, "idx": i, "set": "object"} for i, p in enumerate(object_set)]
    )

    my_steps = args.steps[rank::world]
    print(f"[rank {rank}/{world}] device={device} steps={my_steps}", flush=True)

    for step in my_steps:
        ckpt_path = exp_dir / f"checkpoint-{step}" / "pytorch_lora_weights.safetensors"
        sample_dir = exp_dir / f"checkpoint-{step}" / "sample"

        if not ckpt_path.exists():
            print(f"[rank {rank}] skip step {step}: {ckpt_path} not found", flush=True)
            continue

        sample_dir.mkdir(parents=True, exist_ok=True)

        todo = []
        for pd in prompts:
            for i in range(args.n_gen):
                out = sample_dir / f"{pd['set']}_{pd['idx']}_{i}.png"
                if not out.exists():
                    todo.append((pd, i, out))

        if not todo:
            print(f"[rank {rank}] step {step}: all "
                  f"{len(prompts) * args.n_gen} samples already on disk, skip",
                  flush=True)
            continue
        pipe = StableDiffusionXLPipeline.from_pretrained(
            args.model_id, torch_dtype=torch.float16
        ).to(device)
        pipe.load_lora_weights(str(ckpt_path), weight=1.0)
        pipe.set_progress_bar_config(disable=True)
        print(f"[rank {rank}] step {step}: LoRA loaded, generating "
              f"{len(todo)} of {len(prompts) * args.n_gen} samples "
              f"(rest already on disk)", flush=True)

        for pd, i, out in tqdm(
            todo,
            desc=f"rank {rank} step {step}",
            position=rank,
            leave=True,
        ):
            generator = torch.Generator(device=device).manual_seed(pd["idx"] + i)
            img = pipe(
                pd["prompt"],
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images[0]
            img.save(str(out))

        del pipe
        torch.cuda.empty_cache()

        print(f"[rank {rank}] step {step}: done", flush=True)
    print(f"[rank {rank}] all assigned steps done", flush=True)


if __name__ == "__main__":
    main()
