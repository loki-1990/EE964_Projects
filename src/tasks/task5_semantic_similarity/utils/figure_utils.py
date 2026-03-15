from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image


def list_figures(fig_dir):
    """
    List all figure files in a directory
    """
    fig_dir = Path(fig_dir)
    files = sorted([f for f in fig_dir.iterdir() if f.suffix in [".png", ".jpg", ".jpeg"]])

    for f in files:
        print(f.name)

    return files


def preview_figures(fig_dir, cols=2):
    """
    Show all figures in notebook for quick inspection
    """
    fig_dir = Path(fig_dir)
    files = sorted([f for f in fig_dir.iterdir() if f.suffix in [".png", ".jpg", ".jpeg"]])

    rows = (len(files) + cols - 1) // cols
    plt.figure(figsize=(6 * cols, 4 * rows))

    for i, f in enumerate(files):
        img = Image.open(f)

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(f.name)
        plt.axis("off")

    plt.tight_layout()
    plt.show()