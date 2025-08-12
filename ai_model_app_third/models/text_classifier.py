import os
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


class SimpleTextClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, x):
        return self.net(x)


def _load_embeddings(emb_path):
    """Load embeddings from various formats.
    Returns: (embeddings, paths)
      - embeddings: numpy.ndarray or scipy.sparse matrix
      - paths: list of file paths or None
    Supports:
      - numpy .npz with named arrays 'embeddings' and optional 'paths'
      - scipy.sparse .npz created by scipy.sparse.save_npz
      - plain .npy or .npz with a single unnamed array
    """
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Embeddings file not found: {emb_path}")

    # Try numpy .npz with named arrays first
    try:
        arr = np.load(emb_path, allow_pickle=True)
        # If it's an NpzFile (has .files) and contains 'embeddings'
        if hasattr(arr, 'files') and 'embeddings' in arr.files:
            embeddings = arr['embeddings']
            paths = arr['paths'] if 'paths' in arr.files else None
            return embeddings, paths
        # If it's an NpzFile but without 'embeddings', try infer
        if hasattr(arr, 'files') and len(arr.files) > 0:
            # try first array as embeddings
            first = arr[arr.files[0]]
            # second array might be paths
            paths = None
            if len(arr.files) > 1 and 'paths' in arr.files:
                paths = arr['paths']
            return first, paths
    except Exception:
        # not a standard numpy .npz with named arrays; fall through
        pass

    # Try scipy sparse load (for TF-IDF saved via scipy.sparse.save_npz)
    try:
        import scipy.sparse as sp
        mat = sp.load_npz(emb_path)
        # load meta JSON next to embeddings for paths
        meta_path = emb_path + '.meta.json'
        paths = None
        if os.path.exists(meta_path):
            import json
            with open(meta_path, 'r', encoding='utf-8') as mf:
                meta = json.load(mf)
                paths = meta.get('paths')
        return mat, paths
    except Exception:
        pass

    # Fallback: try to load as a simple numpy array (.npy)
    try:
        arr2 = np.load(emb_path, allow_pickle=True)
        return arr2, None
    except Exception as e:
        raise RuntimeError(f"Unable to load embeddings from {emb_path}: {e}")


def _load_labels_map(labels_csv_path):
    if not labels_csv_path or not os.path.exists(labels_csv_path):
        return None
    df = pd.read_csv(labels_csv_path)
    # Expect columns: filename,label
    if 'filename' not in df.columns or 'label' not in df.columns:
        raise RuntimeError('labels CSV must contain columns: filename,label')
    mapping = {str(row['filename']): row['label'] for _, row in df.iterrows()}
    return mapping


def train_text_classifier(emb_path='docs/embeddings.npz', labels_csv_path='docs/labels.csv',
                          out_model_path='docs/text_model.pth', epochs=20, batch_size=16, lr=1e-3, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    embeddings, paths = _load_embeddings(emb_path)

    # If embeddings is a scipy sparse matrix, convert to dense
    try:
        import scipy.sparse as sp
        if isinstance(embeddings, sp.spmatrix):
            embeddings = embeddings.toarray()
    except Exception:
        pass

    embeddings = np.asarray(embeddings)

    # If embeddings stored as an object array (e.g. array of arrays), try to vstack
    if embeddings.dtype == object:
        try:
            embeddings = np.vstack([np.asarray(x) for x in embeddings])
        except Exception:
            # If it's a single-object wrapping the real array, unwrap
            if embeddings.size == 1 and isinstance(embeddings[0], (np.ndarray, list)):
                embeddings = np.asarray(embeddings[0])
            else:
                raise RuntimeError(
                    "Embeddings file contains an object-dtype array that couldn't be converted to 2D. "
                    "Inspect the embeddings file (np.load(...).files and shapes)."
                )

    # Ensure embeddings are 2D
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    n_samples, emb_dim = embeddings.shape

    # If there are too few samples, provide a clear error
    if n_samples < 2:
        raise RuntimeError(
            f"Not enough documents to train a classifier: n_samples={n_samples}. "
            "Provide at least 2 documents with labels (docs/labels.csv) or add more text files to docs/."
        )

    labels_map = _load_labels_map(labels_csv_path)
    y = None
    if labels_map and paths is not None:
        # Map filenames to labels, using basename matching
        y_list = []
        for p in paths:
            name = os.path.basename(p)
            if name in labels_map:
                y_list.append(labels_map[name])
            else:
                y_list.append(None)
        # filter out None
        valid_idx = [i for i, lab in enumerate(y_list) if lab is not None]
        if len(valid_idx) == 0:
            labels_map = None
        else:
            X = embeddings[valid_idx]
            y_vals = [y_list[i] for i in valid_idx]
            # convert labels to integer indices
            unique = sorted(list(set(y_vals)))
            if len(unique) < 2:
                # only one class present in provided labels — cannot train a classifier
                raise RuntimeError(
                    f"Labels provided map to a single class ({unique[0]}). "
                    "A classifier requires at least 2 distinct classes. Provide labels with >=2 classes."
                )
            label2idx = {lab: idx for idx, lab in enumerate(unique)}
            y = np.array([label2idx[lab] for lab in y_vals])
            paths = [paths[i] for i in valid_idx]
            embeddings = X
            n_samples = embeddings.shape[0]

    # If y wasn't constructed from labels, fall back to clustering if available
    if y is None:
        if SKLEARN_AVAILABLE:
            # ensure n_clusters <= n_samples
            k = min(2, n_samples)
            if k < 2:
                raise RuntimeError(
                    f"Not enough samples for clustering-based pseudo-labels: n_samples={n_samples}. "
                    "Add more documents or provide labels with at least 2 classes."
                )
            kmeans = KMeans(n_clusters=k, random_state=0)
            y = kmeans.fit_predict(embeddings)
            n_classes = k
        else:
            raise RuntimeError('No labels provided or matched and scikit-learn not available for clustering fallback.')
    else:
        n_classes = len(set(y.tolist()))

    # Convert to tensors
    X_tensor = torch.tensor(embeddings, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleTextClassifier(input_dim=emb_dim, hidden_dim=128, n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'loss': [], 'acc': []}
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
        avg_loss = total_loss / len(loader.dataset)
        acc = correct / len(loader.dataset)
        history['loss'].append(avg_loss)
        history['acc'].append(acc)
        print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} - acc: {acc:.4f}")

    # Save model and metadata
    os.makedirs(os.path.dirname(out_model_path), exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'input_dim': emb_dim, 'n_classes': n_classes}, out_model_path)

    return {'model_path': out_model_path, 'n_samples': n_samples, 'n_classes': n_classes, 'history': history}
