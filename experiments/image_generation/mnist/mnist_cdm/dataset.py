"""Rotated-MNIST dataset construction, and the classifier it depends on.

Ported from `MNIST/src/dataset.py` and `MNIST/train/train_uncond.py` of
`orineo1/conditional-matching-paper` (branch `main`, MNIST tree at 7c856e2).

Two different datasets are built here, and they are not interchangeable:

  * `AugmentedMNISTDataset` -- the SEARCH set the conditional iCT angle model is
    trained on. Built from the 60,000 MNIST training images; each image is also
    kept at 90/180/270 degrees when `ConstructionCNN` still predicts the same digit
    with confidence above `confidence_threshold`, and every retained entry is
    then rotated by a further uniform angle.
  * `RotatedMNISTDataset` -- the PRIOR set the unconditional DDPM is trained on.
    All 70,000 MNIST images (train + test), each given an independent uniform
    rotation at access time, with no classifier filtering.

`ConstructionCNN` / `train_classifier` here are the dataset-construction classifier.
The rotation-robust classifier used to LABEL generated digits is a different
network with different training, and lives in `mnist_cdm/classifier.py`.

Exports:
    - ConstructionCNN, train_classifier      (dataset-construction classifier)
    - AugmentedMNISTDataset                  (conditional-model training set)
    - RotatedMNISTDataset                    (prior training set)
"""

import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from PIL import Image
from tqdm import tqdm


NORM_MEAN       = 0.1307
NORM_STD        = 0.3081
ROTATION_ANGLES = [90, 180, 270]


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class ConstructionCNN(nn.Module):
    """Three-block CNN classifier used to BUILD the dataset.

    Named `ImprovedCNN` in the source repo, where a second and different class
    of the same name lived in `MNIST/src/classifier.py` (the rotation-robust
    evaluation classifier, ported to `mnist_cdm/classifier.py`). The two are not
    interchangeable -- this one has no input/mid dropout and is trained without
    label smoothing or noise augmentation -- so the name is made distinct here.
    A class rename does not affect state-dict keys.
    """

    def __init__(self):
        super().__init__()
        self.conv1       = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1         = nn.BatchNorm2d(32)
        self.conv2       = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2         = nn.BatchNorm2d(32)
        self.pool1       = nn.MaxPool2d(2)
        self.dropout1    = nn.Dropout2d(0.25)

        self.conv3       = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3         = nn.BatchNorm2d(64)
        self.conv4       = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4         = nn.BatchNorm2d(64)
        self.pool2       = nn.MaxPool2d(2)
        self.dropout2    = nn.Dropout2d(0.25)

        self.conv5       = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5         = nn.BatchNorm2d(128)
        self.pool3       = nn.MaxPool2d(2)
        self.dropout3    = nn.Dropout2d(0.25)

        self.fc1         = nn.Linear(128 * 3 * 3, 256)
        self.bn_fc1      = nn.BatchNorm1d(256)
        self.dropout_fc1 = nn.Dropout(0.5)
        self.fc2         = nn.Linear(256, 128)
        self.bn_fc2      = nn.BatchNorm1d(128)
        self.dropout_fc2 = nn.Dropout(0.5)
        self.fc3         = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))))
        x = self.dropout1(x)
        x = self.pool2(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(x)))))))
        x = self.dropout2(x)
        x = self.pool3(F.relu(self.bn5(self.conv5(x))))
        x = self.dropout3(x)
        x = x.view(-1, 128 * 3 * 3)
        x = self.dropout_fc1(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.dropout_fc2(F.relu(self.bn_fc2(self.fc2(x))))
        return self.fc3(x)


def train_classifier(epochs=15, batch_size=128, lr=1e-3, device='cpu',
                     save_path='best_mnist_classifier.pth', root='./data',
                     limit=None):
    """Train ConstructionCNN on MNIST and return the best model."""
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((NORM_MEAN,), (NORM_STD,)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((NORM_MEAN,), (NORM_STD,)),
    ])

    train_set = datasets.MNIST(root, train=True, download=True, transform=train_transform)
    test_set = datasets.MNIST(root, train=False, download=True, transform=test_transform)
    if limit:                      # smoke tier only
        train_set = torch.utils.data.Subset(train_set, range(limit))
        test_set = torch.utils.data.Subset(test_set, range(limit))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    model     = ConstructionCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()
    best_acc  = 0.0

    print(f"Training construction classifier — {sum(p.numel() for p in model.parameters()):,} parameters")
    for epoch in range(1, epochs + 1):
        model.train()
        correct, total = 0, 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out  = model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            correct += out.argmax(1).eq(target).sum().item()
            total   += target.size(0)

        model.eval()
        test_loss, test_correct, test_total = 0, 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                out        = model(data)
                test_loss += criterion(out, target).item()
                test_correct += out.argmax(1).eq(target).sum().item()
                test_total   += target.size(0)

        test_acc = 100. * test_correct / max(test_total, 1)
        scheduler.step(test_loss / len(test_loader))
        print(f"Epoch {epoch:2d}/{epochs} | Train {100.*correct/total:.2f}% | Test {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(f"  → New best: {best_acc:.2f}%")

    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    print(f"Classifier ready — best test accuracy: {best_acc:.2f}%")
    return model


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _rotate_image(tensor_image, angle, device):
    """Rotate a normalised (C,H,W) tensor by `angle` degrees, return normalised tensor."""
    denorm  = tensor_image.cpu() * NORM_STD + NORM_MEAN
    pil_img = transforms.ToPILImage()(denorm.squeeze(0).clamp(0, 1))
    rotated = pil_img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
    tensor  = transforms.ToTensor()(rotated)
    tensor  = (tensor - NORM_MEAN) / NORM_STD
    return tensor.to(device)


class AugmentedMNISTDataset(Dataset):
    """
    MNIST dataset augmented with rotationally ambiguous copies.

    For each base image:
      - Always included at logical angle 0°.
      - Also included at 90°, 180°, 270° if the classifier predicts the same
        digit with confidence >= threshold when shown the rotated image.
      - A final uniform random rotation in [0°, 360°) is applied to every
        entry; the stored angle is (logical_angle + uniform_angle) % 360.

    Returns: (image, label, total_angle)
    """

    def __init__(self, classifier_model, confidence_threshold=0.9999, device='cpu',
                 root='./data', limit=None):
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((NORM_MEAN,), (NORM_STD,)),
        ])
        base_data = datasets.MNIST(root, train=True, download=True, transform=base_transform)
        if limit:                  # smoke tier only
            base_data = torch.utils.data.Subset(base_data, range(limit))

        self.final_images = []
        self.final_labels = []
        self.final_angles = []

        # --- Stage 1: collect logical-angle entries ---
        print("Stage 1: collecting logical-angle entries …")
        temp_entries = []  # (image_tensor, label, logical_angle_deg)

        for image, label in tqdm(base_data, desc="Base images"):
            base_image = image.to(device)
            temp_entries.append((base_image, label, 0.0))  # angle=0 always kept

            for angle in ROTATION_ANGLES:
                rotated = _rotate_image(base_image, angle, device)
                with torch.no_grad():
                    probs    = torch.softmax(classifier_model(rotated.unsqueeze(0)), dim=1)
                    max_prob, _ = probs.max(1)

                if max_prob.item() > confidence_threshold:
                    temp_entries.append((base_image, label, float(angle)))

        print(f"Total logical entries before final rotation: {len(temp_entries)}")

        # --- Stage 2: apply final uniform random rotation ---
        print("Stage 2: applying final uniform rotation …")
        for base_image, label, logical_angle in temp_entries:
            uniform_angle = np.random.uniform(0, 360)
            total_angle   = (logical_angle + uniform_angle) % 360.0
            final_image   = _rotate_image(base_image, uniform_angle, device)
            self.final_images.append(final_image)
            self.final_labels.append(label)
            self.final_angles.append(torch.tensor(total_angle, dtype=torch.float32))

        print(f"Dataset ready — {len(self.final_images)} samples.")

    def __len__(self):
        return len(self.final_images)

    def __getitem__(self, idx):
        return self.final_images[idx], self.final_labels[idx], self.final_angles[idx]


class RotatedMNISTDataset(Dataset):
    """MNIST train + test, with a uniform random rotation sampled per access.

    The training set of the unconditional DDPM prior. No classifier filtering:
    every one of the 70,000 images is kept.
    """

    def __init__(self, root='./data', download=True):
        train_data = datasets.MNIST(root, train=True, download=download,
                                    transform=transforms.ToTensor())
        test_data = datasets.MNIST(root, train=False, download=download,
                                   transform=transforms.ToTensor())
        self.data = torch.utils.data.ConcatDataset([train_data, test_data])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        rotation = np.random.uniform(0, 360)
        image = TF.rotate(image.float(), rotation, fill=0)
        return image, label, torch.tensor(rotation, dtype=torch.float32)
