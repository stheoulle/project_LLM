import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn.functional as F

def resize_volume(volume, target_shape=(128, 224, 224)):
    # volume: [C, D, H, W]
    volume = volume.unsqueeze(0)  # [1, C, D, H, W]
    resized = F.interpolate(volume, size=target_shape, mode='trilinear', align_corners=False)
    return resized.squeeze(0)  # [C, D, H, W]


class MultimodalDataset(Dataset):
    def __init__(self, images, text, tabular, labels, target_depth=128):
        self.images = [resize_volume(img, target_depth) for img in images]
        self.text = text
        self.tabular = tabular
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.text[idx], self.tabular[idx], self.labels[idx]

def train_model(model, data, epochs=5, target_depth=128):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    history = {'loss': [], 'acc': []}

    if 'text' in data and 'tabular' in data:
        # 🧠 Multimodal case
        dataset = MultimodalDataset(data['images'], data['text'], data['tabular'], data['labels'], target_depth)
    else:
        # 🖼️ Image only case
        resized_images = [resize_volume(img, (128, 224, 224)) for img in data['images']]
        inputs = torch.stack(resized_images)
        dataset = TensorDataset(inputs, data['labels'])

    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0, 0

        for batch in loader:
            if 'text' in data:
                image, text, tabular, y = batch
                out = model(image, text, tabular)
            else:
                x, y = batch
                out = model(x)

            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()

        acc = correct / len(loader.dataset)
        history['loss'].append(total_loss)
        history['acc'].append(acc)
        print(f"Epoch {epoch+1}: loss={total_loss:.4f}, acc={acc:.4f}")

    return history
