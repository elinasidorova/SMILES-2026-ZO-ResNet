"""
head_init.py – Initialize CIFAR100 head by mapping to semantically
closest ImageNet-1k classes via sentence embeddings.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from sentence_transformers import SentenceTransformer
import json
import os

_MAPPING_CACHE_FILE = "/content/imagenet_to_cifar100_mapping.pt"

def _get_class_names():
    """Return lists of ImageNet-1k and CIFAR100 class names."""
    cifar100_classes = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
        'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree',
        'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
        'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm'
    ]

    import urllib.request
    url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    urllib.request.urlretrieve(url, "imagenet_class_index.json")
    with open("imagenet_class_index.json", 'r') as f:
        imagenet_idx = json.load(f)
    imagenet_classes = [imagenet_idx[str(i)][1] for i in range(1000)]
    return imagenet_classes, cifar100_classes

def _build_mapping():
    """Build a tensor of shape (100,) with indices of best ImageNet class for each CIFAR100 class."""
    if os.path.exists(_MAPPING_CACHE_FILE):
        return torch.load(_MAPPING_CACHE_FILE, map_location='cpu')

    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    imagenet_classes, cifar100_classes = _get_class_names()

    imagenet_emb = st_model.encode(imagenet_classes, convert_to_tensor=True, show_progress_bar=False)
    cifar100_emb = st_model.encode(cifar100_classes, convert_to_tensor=True, show_progress_bar=False)

    imagenet_emb = imagenet_emb / imagenet_emb.norm(dim=1, keepdim=True)
    cifar100_emb = cifar100_emb / cifar100_emb.norm(dim=1, keepdim=True)
    similarity = cifar100_emb @ imagenet_emb.T
    best_indices = similarity.argmax(dim=1).cpu()

    torch.save(best_indices, _MAPPING_CACHE_FILE)
    return best_indices

def init_last_layer(layer: nn.Linear) -> None:
    """Initialize the new 100-class head with weights from semantically
    closest ImageNet-1k classes."""
    print("[head_init] Loading original ResNet18 for weight transfer...")
    original_model = models.resnet18(weights="IMAGENET1K_V1")
    old_weight = original_model.fc.weight.data  # (1000, 512)
    old_bias = original_model.fc.bias.data
    del original_model

    indices = _build_mapping()
    indices = indices.to(old_weight.device)

    new_weight = old_weight[indices].clone()
    new_bias = old_bias[indices].clone()

    scale = (1000 / 100) ** 0.5
    new_weight *= scale
    new_bias *= scale

    with torch.no_grad():
        layer.weight.data = new_weight.to(layer.weight.device)
        layer.bias.data = new_bias.to(layer.bias.device)

    print("[head_init] Initialized with semantically mapped ImageNet weights.")
