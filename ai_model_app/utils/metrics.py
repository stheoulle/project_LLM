import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import streamlit as st

def evaluate_model(model, data):
    model.eval()
    inputs = torch.stack(data['images'])
    labels = data['labels']
    with torch.no_grad():
        preds = model(inputs).argmax(1)
    report = classification_report(labels, preds, output_dict=True)
    return {"y_true": labels, "y_pred": preds, "report": report}

def plot_metrics(history, results, streamlit=False):
    fig, ax = plt.subplots()
    ax.plot(history['loss'], label='Loss')
    ax.plot(history['acc'], label='Accuracy')
    ax.set_title("Courbes d’entraînement")
    ax.legend()

    if streamlit:
        st.pyplot(fig)
        st.json(results['report'])
    else:
        plt.show()
