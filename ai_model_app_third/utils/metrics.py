import torch
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np

def resize_volume(volume, target_depth=128):
    volume = volume.unsqueeze(0)  # [1, C, D, H, W]
    resized = F.interpolate(volume, size=(target_depth, volume.shape[2], volume.shape[3]),
                            mode='trilinear', align_corners=False)
    return resized.squeeze(0)


def evaluate_model(model, data, target_depth=128, batch_size=32):
    """Evaluate a model on image, multimodal or text embeddings data.

    Supported data shapes:
      - Image-only: data contains 'images' (list of tensors) and 'labels'
      - Multimodal: data contains 'images','text','tabular','labels'
      - Text-only: data contains 'embeddings' (numpy or tensor) and 'labels'

    Returns dict with y_true (tensor), y_pred (tensor), report (sklearn dict), confusion_matrix (np.array)
    """
    model.eval()
    y_true = []
    y_pred = []

    # Text-only evaluation (embeddings + labels)
    if 'embeddings' in data:
        X = data['embeddings']
        Y = data.get('labels')
        if Y is None:
            raise RuntimeError('Text evaluation requires labels in data["labels"]')

        # Convert to torch tensors
        if isinstance(X, np.ndarray):
            X_tensor = torch.tensor(X, dtype=torch.float32)
        elif torch.is_tensor(X):
            X_tensor = X.float()
        else:
            X_tensor = torch.tensor(np.asarray(X), dtype=torch.float32)

        if isinstance(Y, np.ndarray):
            y_list = Y.tolist()
        elif torch.is_tensor(Y):
            y_list = Y.cpu().numpy().tolist()
        else:
            y_list = list(Y)

        # Send model and data to same device
        try:
            device = next(model.parameters()).device
        except Exception:
            device = torch.device('cpu')
        X_tensor = X_tensor.to(device)

        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                xb = X_tensor[i:i+batch_size]
                out = model(xb)
                preds = out.argmax(dim=1).cpu().numpy().tolist()
                y_pred.extend(preds)

        y_true = [int(v) for v in y_list]
        y_pred = [int(v) for v in y_pred]

    else:
        # Image-only or multimodal evaluation (existing behaviour)
        if 'text' in data and 'tabular' in data:
            multimodal = True
            samples = zip(data['images'], data['text'], data['tabular'], data['labels'])
        else:
            multimodal = False
            samples = zip(data['images'], data['labels'])

        with torch.no_grad():
            for sample in samples:
                if multimodal:
                    img, txt, tab, label = sample
                    img = resize_volume(img, target_depth).unsqueeze(0)  # Add batch dim
                    txt = txt.unsqueeze(0)
                    tab = tab.unsqueeze(0)
                    pred = model(img, txt, tab).argmax(1)
                else:
                    img, label = sample
                    img = resize_volume(img, target_depth).unsqueeze(0)
                    pred = model(img).argmax(1)

                y_true.append(label.item() if torch.is_tensor(label) else label)
                y_pred.append(pred.item())

    # Ensure tensors for results
    y_true_tensor = torch.tensor(y_true)
    y_pred_tensor = torch.tensor(y_pred)

    report = classification_report(y_true, y_pred, output_dict=True)
    matrix = confusion_matrix(y_true, y_pred)

    return {
        "y_true": y_true_tensor,
        "y_pred": y_pred_tensor,
        "report": report,
        "confusion_matrix": matrix
    }


def plot_metrics(history, results, streamlit=False):
    # Plot training loss/accuracy if available
    if history is not None and isinstance(history, dict) and 'loss' in history and 'acc' in history:
        fig, ax = plt.subplots()
        ax.plot(history['loss'], label='Loss')
        ax.plot(history['acc'], label='Accuracy')
        ax.set_title("Courbes d’entraînement")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Valeur")
        ax.legend()
    else:
        fig = None

    # Confusion matrix and report
    y_true = results['y_true'].cpu().numpy() if torch.is_tensor(results['y_true']) else np.asarray(results['y_true'])
    y_pred = results['y_pred'].cpu().numpy() if torch.is_tensor(results['y_pred']) else np.asarray(results['y_pred'])
    cm = confusion_matrix(y_true, y_pred)
    cm_fig, cm_ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=cm_ax)
    cm_ax.set_title("Matrice de confusion")
    cm_ax.set_xlabel("Prédiction")
    cm_ax.set_ylabel("Réel")

    if streamlit:
        if fig is not None:
            st.pyplot(fig)
        st.pyplot(cm_fig)
        st.json(results['report'])
    else:
        if fig is not None:
            plt.show()
        plt.show()
        print("📊 Rapport de classification :")
        for label, metrics in results['report'].items():
            if isinstance(metrics, dict):
                print(f"{label}: precision={metrics['precision']:.2f}, recall={metrics['recall']:.2f}, f1={metrics['f1-score']:.2f}")
