"""Download Imagenette and build a small, preprocessed test set (data.pt).

Imagenette has 10 easy ImageNet classes. Each one maps to a class number in the
1000 class ImageNet list that ResNet-18 was trained on, so we can check the
model's answers without any retraining.
"""
import os
import random
import tarfile
import urllib.request

import torch
from PIL import Image
from torchvision import transforms

URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
FOLDER = "imagenette2-320"
PER_CLASS = 30   # 10 classes x 30 = 300 images

# Imagenette folder name -> ImageNet-1k class index
WNID_TO_IDX = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}

if not os.path.isdir(FOLDER):
    print("Downloading Imagenette (about 325 MB), please wait...")
    urllib.request.urlretrieve(URL, "imagenette.tgz")
    with tarfile.open("imagenette.tgz") as t:
        t.extractall()
    os.remove("imagenette.tgz")

# The same preprocessing ResNet-18 expects
prep = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

rng = random.Random(0)   # fixed seed so everyone gets the same 300 images
items = []
for wnid, idx in WNID_TO_IDX.items():
    d = os.path.join(FOLDER, "val", wnid)
    files = sorted(os.listdir(d))
    for f in rng.sample(files, PER_CLASS):
        items.append((os.path.join(d, f), idx))
rng.shuffle(items)   # mix the classes

x = torch.stack([prep(Image.open(p).convert("RGB")) for p, _ in items])
y = torch.tensor([idx for _, idx in items])
torch.save({"x": x, "y": y}, "data.pt")
print("Saved data.pt:", tuple(x.shape), "labels:", tuple(y.shape))
