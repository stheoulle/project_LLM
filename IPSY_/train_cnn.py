import os
import argparse
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import DBTDataset
from model import build_model


def collate_fn(batch):
    imgs = [torch.tensor(b[0]) if not isinstance(b[0], torch.Tensor) else b[0] for b in batch]
    imgs = torch.stack(imgs)
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return imgs, labels


def prepare_splits(labels_csv, filepaths_csv, test_size=0.2, val_size=0.1, random_state=42):
    df_labels = pd.read_csv(labels_csv)
    df_paths = pd.read_csv(filepaths_csv)
    merged = pd.merge(df_labels, df_paths, on=["PatientID", "StudyUID", "View"]) 
    merged["label_idx"] = merged[["Normal", "Actionable", "Benign", "Cancer"]].values.argmax(axis=1)

    print("Label distribution:", Counter(merged["label_idx"]))
    train_val, test = train_test_split(
        merged, test_size=test_size, stratify=merged["label_idx"], random_state=random_state
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size / (1 - test_size),
        stratify=train_val["label_idx"],
        random_state=random_state,
    )
    print(f"Train size: {len(train)}, Val size: {len(val)}, Test size: {len(test)}")
    train.to_csv("train_split.csv", index=False)
    val.to_csv("val_split.csv", index=False)
    test.to_csv("test_split.csv", index=False)
    return "train_split.csv", "val_split.csv", "test_split.csv"


def evaluate(model, loader, device, class_names):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, dtype=torch.float32)
            labels = labels.to(device)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Check if model exists
    model_path = "IPSY_/best_model.pth"
    if os.path.exists(model_path):
        print(f"Model found at {model_path}, skipping training...")
        _, _, test_csv = prepare_splits(args.labels, args.filepaths, test_size=args.test_size, val_size=args.val_size)

        test_ds = DBTDataset(test_csv, args.filepaths, root=args.data_root, verbose=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)

        model = build_model(n_classes=4).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from epoch {checkpoint['epoch']} with val_acc={checkpoint['val_acc']:.4f}")

        evaluate(model, test_loader, device, class_names=["Normal", "Actionable", "Benign", "Cancer"])
        return

    # If no model → train
    train_csv, val_csv, test_csv = prepare_splits(args.labels, args.filepaths, test_size=args.test_size, val_size=args.val_size)

    train_ds = DBTDataset(train_csv, args.filepaths, root=args.data_root, verbose=True)
    val_ds = DBTDataset(val_csv, args.filepaths, root=args.data_root, verbose=True)
    test_ds = DBTDataset(test_csv, args.filepaths, root=args.data_root, verbose=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)

    model = build_model(n_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device, dtype=torch.float32)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # validation
        model.eval()
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, dtype=torch.float32)
                labels = labels.to(device)
                outputs = model(imgs)
                _, preds = outputs.max(1)
                v_correct += (preds == labels).sum().item()
                v_total += labels.size(0)
        val_acc = v_correct / v_total if v_total > 0 else 0.0

        print(f"Epoch {epoch}/{args.epochs} - train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_acc": val_acc}, model_path)

    print("Training finished. Best val acc:", best_val_acc)

    # Final evaluation on test set
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    evaluate(model, test_loader, device, class_names=["Normal", "Actionable", "Benign", "Cancer"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="labels csv path")
    parser.add_argument("--filepaths", required=True, help="filepaths csv path")
    parser.add_argument("--data-root", default=".", help="root directory to prepend to classic_path entries")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    args = parser.parse_args()

    train(args)
