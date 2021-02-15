"""Supportive functions for computing and presenting results. Minimum of app logic here."""
import io

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def normalize(vector):
    """Vector normalization."""
    norm = np.linalg.norm(vector)

    if norm == 0:
        norm = np.finfo(vector.dtype).eps

    return vector / norm


def plot_to_image(fig):
    """Converts given figure into image object."""
    buf = io.BytesIO()

    fig.savefig(buf)
    buf.seek(0)

    return Image.open(buf)


def similarity_plot(similaity_matrix, artists_names):
    """Plot similarity matrix """
    fig, ax = plt.subplots(figsize=(10, 10))

    sns.heatmap(similaity_matrix,
                xticklabels=artists_names,
                yticklabels=artists_names,
                square=True,
                annot=True)
    ax.set_title('Artists similarity')

    return fig