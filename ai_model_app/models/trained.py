import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

def resize_volume(volume, target_depth=128):
    # volume: [C, D, H, W]
    volume = volume.unsqueeze(0)  # [1, C, D, H, W]
    resized = F.interpolate(volume, size=(target_depth, volume.shape[3], volume.shape[4]),
                            mode='trilinear', align_corners=False)
    return resized.squeeze(0)  # [C, target_depth, H, W]

def train_model(model, data, epochs=5, target_depth=128):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Resize all volumes to have same depth
    resized_images = [resize_volume(img, target_depth) for img in data['images']]
    inputs = torch.stack(resized_images)  # Now all have same shape

    labels = data['labels']
    
    dataset = TensorDataset(inputs, labels)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    history = {'loss': [], 'acc': []}
    
    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0, 0
        for x, y in loader:
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
