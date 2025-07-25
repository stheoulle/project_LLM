import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

def train_model(model, data, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    inputs = torch.stack(data['images'])
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
