import torch
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from torch.utils.data import DataLoader, Dataset

def resize_volume(volume, target_depth=128):
    volume = volume.unsqueeze(0)  # [1, C, D, H, W]
    resized = F.interpolate(volume, size=(target_depth, volume.shape[2], volume.shape[3]),
                            mode='trilinear', align_corners=False)
    return resized.squeeze(0)

def evaluate_model(model, data, target_depth=128):
    model.eval()
    y_true, y_pred = [], []

    # ✅ New: Support torch Dataset yielding (image, tabular, label) or (image, text, tabular, label)
    if isinstance(data, Dataset):
        loader = DataLoader(data, batch_size=32, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 4:
                    image, text, tabular, labels = batch
                else:
                    image, tabular, labels = batch
                    text = None
                # Try multimodal call, otherwise fallback to image-only
                try:
                    if text is not None:
                        out = model(image, tabular=tabular, text=text)
                    else:
                        out = model(image, tabular=tabular)
                except TypeError:
                    out = model(image)
                preds = out.argmax(1)
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        matrix = confusion_matrix(y_true, y_pred)
        return {
            "y_true": torch.tensor(y_true),
            "y_pred": torch.tensor(y_pred),
            "report": report,
            "confusion_matrix": matrix
        }

    # 🔙 Legacy dict-based evaluation
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
                pred = model(img, tabular=tab, text=txt).argmax(1)
            else:
                img, label = sample
                img = resize_volume(img, target_depth).unsqueeze(0)
                pred = model(img).argmax(1)

            y_true.append(label.item() if torch.is_tensor(label) else label)
            y_pred.append(pred.item())

    y_true_tensor = torch.tensor(y_true)
    y_pred_tensor = torch.tensor(y_pred)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred)

    return {
        "y_true": y_true_tensor,
        "y_pred": y_pred_tensor,
        "report": report,
        "confusion_matrix": matrix
    }

def plot_metrics(history, results, streamlit=False):
    # 📈 Courbes de perte et d'accuracy
    fig, ax = plt.subplots()
    ax.plot(history['loss'], label='Loss')
    ax.plot(history['acc'], label='Accuracy')
    ax.set_title("Courbes d’entraînement")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Valeur")
    ax.legend()

    # 📊 Matrice de confusion
    y_true = results['y_true'].cpu().numpy()
    y_pred = results['y_pred'].cpu().numpy()
    cm = confusion_matrix(y_true, y_pred)
    cm_fig, cm_ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=cm_ax)
    cm_ax.set_title("Matrice de confusion")
    cm_ax.set_xlabel("Prédiction")
    cm_ax.set_ylabel("Réel")

    if streamlit:
        st.pyplot(fig)
        st.pyplot(cm_fig)
        st.json(results['report'])
    else:
        plt.show()  # affiche courbes + matrice
        print("📊 Rapport de classification :")
        for label, metrics in results['report'].items():
            if isinstance(metrics, dict):
                print(f"{label}: precision={metrics['precision']:.2f}, recall={metrics['recall']:.2f}, f1={metrics['f1-score']:.2f}")
