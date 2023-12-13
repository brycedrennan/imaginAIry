import torch
from torch.cuda import OutOfMemoryError

from imaginairy.api import imagine_image_files
from imaginairy.schema import ImaginePrompt
from imaginairy.utils import get_device


def assess_memory_usage():
    assert get_device() == "cuda"
    img_size = 3048
    prompt = ImaginePrompt("strawberries", size=64, seed=1)
    imagine_image_files([prompt], outdir="outputs")
    datalog = []
    while True:
        torch.cuda.reset_peak_memory_stats()
        prompt = ImaginePrompt(
            "beautiful landscape, Unreal Engine 5, RTX, AAA Game, Detailed 3D Render, Cinema4D",
            size=img_size,
            seed=1,
            steps=2,
        )
        try:
            imagine_image_files([prompt], outdir="outputs")
        except OutOfMemoryError as e:
            print(f"Out of memory at {img_size}x{img_size} size image.")
            print(e)
            break
        max_used = torch.cuda.max_memory_allocated() / 1024**3
        datalog.append((img_size, max_used))
        print(f"{img_size},{max_used:.2f}\n")
        img_size += 128

    with open("img_size_memory_usage.csv", "w", encoding="utf-8") as f:
        f.write("img_size,max_used\n")
        for img_size, max_used in datalog:
            f.write(f"{img_size},{max_used:.2f}\n")


if __name__ == "__main__":
    assess_memory_usage()
