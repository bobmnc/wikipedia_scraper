import matplotlib.pyplot as plt
import umap
import numpy as np

def visualize_dataset(embeddings : list[np.ndarray],labels : list):
    umap_embedder = umap.UMAP()
    out = umap_embedder.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))

    # Scatter plot with different colors for each label
    unique_labels = list(set(labels))
    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(out[indices, 0], out[indices, 1], label=label)

    plt.title('UMAP Visualization of Embeddings')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend()
    plt.show()

