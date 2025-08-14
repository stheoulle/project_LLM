"""
CDD_CNN_Transformer_best.py

Best-practice multimodal CNN+Transformer training script adapted from the
CDD_CNN_Transformer notebooks in the workspace.

- Robust image loader that accepts partial/missing views (fills with zeros)
- Optional RadImageNet DenseNet121 encoder if weights available; fallback to small CNN
- TimeDistributed encoder over 8 views, Transformer sequence aggregator
- TF-IDF text features, standardized metadata with one-hot encoding
- Class weighting, focal loss option, callbacks (EarlyStopping, ModelCheckpoint, CSVLogger)
- Save/restore extracted image features to speed up iterative experiments

Usage: run from project root. Adjust paths and hyperparameters in the CONFIG section.
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Dropout, Conv2D, MaxPooling2D,
                                     GlobalAveragePooling2D, GlobalAveragePooling1D,
                                     TimeDistributed, Concatenate, Lambda)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from PIL import Image

# ---------------- CONFIG ----------------
CONFIG = {
    "image_dir_le": "CDD-CESM/PKG - CDD-CESM/CDD-CESM/Low energy images of CDD-CESM",
    "image_dir_sub": "CDD-CESM/PKG - CDD-CESM/CDD-CESM/Subtracted images of CDD-CESM",
    "json_dir": "CDD-CESM/json_output",
    "excel_path": "processed_metadata.csv",
    "img_size": (224, 224),
    "n_views": 8,
    "tfidf_max_features": 1000,
    "batch_size": 16,
    "epochs": 30,
    "seed": 42,
    "radimagenet_weights": "weights/RadImageNet-DenseNet121_notop.h5",  # may not exist
    "save_features_dir": "model",
    "features_names": ("X_img_train_feats.npy", "X_img_val_feats.npy", "X_img_test_feats.npy"),
    "model_ckpt": "best_model_phase2.h5",
    "metrics_log": "model_metrics_log.csv",
}

os.makedirs(CONFIG['save_features_dir'], exist_ok=True)
np.random.seed(CONFIG['seed'])

# --------------- Utilities ----------------

def load_metadata(excel_path):
    df = pd.read_csv(excel_path)
    df = df.dropna(subset=['Patient_ID', 'Pathology Classification/ Follow up'])
    df['Patient_ID'] = df['Patient_ID'].astype(str)
    return df


def load_texts(df, json_dir):
    texts = []
    for pid in df['Patient_ID']:
        path = os.path.join(json_dir, f"P{pid}.json")
        if os.path.exists(path):
            try:
                with open(path, encoding='utf-8') as f:
                    d = json.load(f)
                flat = []
                for v in d.values():
                    flat.extend(map(str, v) if isinstance(v, list) else [str(v)])
                texts.append(" ".join(flat))
            except Exception:
                texts.append("")
        else:
            texts.append("")
    return texts


# Robust loader: returns (n_views, H, W, 1) with missing images filled by zeros.
# If no view exists for the patient returns None.
def load_images_sequential_partial(patient_id, cfg=CONFIG):
    image_dir_le = cfg['image_dir_le']
    image_dir_sub = cfg['image_dir_sub']
    H, W = cfg['img_size']
    paths = [
        f"{image_dir_le}/P{patient_id}_L_DM_CC.jpg",
        f"{image_dir_le}/P{patient_id}_L_DM_MLO.jpg",
        f"{image_dir_sub}/P{patient_id}_L_CM_CC.jpg",
        f"{image_dir_sub}/P{patient_id}_L_CM_MLO.jpg",
        f"{image_dir_le}/P{patient_id}_R_DM_CC.jpg",
        f"{image_dir_le}/P{patient_id}_R_DM_MLO.jpg",
        f"{image_dir_sub}/P{patient_id}_R_CM_CC.jpg",
        f"{image_dir_sub}/P{patient_id}_R_CM_MLO.jpg",
    ]

    imgs = []
    found_any = False
    for p in paths[:cfg['n_views']]:
        if os.path.exists(p):
            try:
                img = Image.open(p).convert('L').resize((W, H))
                arr = np.array(img, dtype=np.float32) / 255.0
                arr = arr[..., np.newaxis]
                found_any = True
            except Exception:
                arr = np.zeros((H, W, 1), dtype=np.float32)
        else:
            arr = np.zeros((H, W, 1), dtype=np.float32)
        imgs.append(arr)

    return np.stack(imgs, axis=0) if found_any else None


# ----------------- Model pieces -----------------

def focal_loss(gamma=2., alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    return loss_fn


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim),
        ])
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.do1 = Dropout(rate)
        self.do2 = Dropout(rate)

    def call(self, inputs, training=None):
        attn = self.att(inputs, inputs)
        attn = self.do1(attn, training=training)
        out1 = self.ln1(inputs + attn)
        ffn = self.ffn(out1)
        ffn = self.do2(ffn, training=training)
        return self.ln2(out1 + ffn)


def create_radimagenet_encoder(cfg=CONFIG, output_dim=64):
    weight_path = cfg['radimagenet_weights']
    try:
        base = DenseNet121(include_top=False, weights=None, input_shape=(cfg['img_size'][0], cfg['img_size'][1], 3))
        if os.path.exists(weight_path):
            base.load_weights(weight_path)
            base.trainable = False
            print("Loaded RadImageNet weights.")
        else:
            raise FileNotFoundError
        inp = Input(shape=(cfg['img_size'][0], cfg['img_size'][1], 1))
        x = Lambda(lambda img: tf.image.grayscale_to_rgb(img))(inp)
        x = preprocess_input(x)
        x = base(x, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(output_dim, activation='relu')(x)
        return Model(inp, x, name='radimage_encoder')
    except Exception:
        print("RadImageNet not available or failed to load -> using small CNN encoder fallback.")
        return create_small_cnn_encoder(cfg, output_dim)


def create_small_cnn_encoder(cfg=CONFIG, output_dim=64):
    inp = Input(shape=(cfg['img_size'][0], cfg['img_size'][1], 1))
    x = Conv2D(32, 3, activation='relu', padding='same')(inp)
    x = MaxPooling2D(2)(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(2)(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(output_dim, activation='relu')(x)
    return Model(inp, x, name='small_cnn_encoder')


def build_multimodal_model(img_feat_dim, meta_dim, text_dim, n_classes, cfg=CONFIG):
    # Inputs
    img_feat_input = Input(shape=(img_feat_dim,), name='img_feat')
    meta_input = Input(shape=(meta_dim,), name='meta_input')
    text_input = Input(shape=(text_dim,), name='text_input')

    # Simple fusion + transformer on fused vector
    x_meta = Dense(64, activation='relu')(meta_input)
    x_text = Dense(64, activation='relu')(text_input)

    x = Concatenate()([img_feat_input, x_meta, x_text])
    # make sequence length 1 to use transformer block (works as cross-modal mixing)
    x_seq = Lambda(lambda z: tf.expand_dims(z, axis=1))(x)
    x_seq = TransformerBlock(embed_dim=int(x.shape[-1]), num_heads=4, ff_dim=128)(x_seq)
    x = GlobalAveragePooling1D()(x_seq)

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=[img_feat_input, meta_input, text_input], outputs=out, name='multimodal_phase2')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=focal_loss(), metrics=['accuracy'])
    return model


# ---------------- Pipeline ----------------

def prepare_dataset(cfg=CONFIG):
    df = load_metadata(cfg['excel_path'])
    texts = load_texts(df, cfg['json_dir'])

    sampled_df = df.sample(frac=0.8, random_state=cfg['seed'])
    sampled_patient_ids = sampled_df['Patient_ID'].tolist()

    images = []
    labels = []
    pids = []
    skipped = 0
    for pid, label in zip(sampled_df['Patient_ID'], sampled_df['Pathology Classification/ Follow up']):
        seq = load_images_sequential_partial(pid, cfg)
        if seq is not None:
            images.append(seq)
            labels.append(label)
            pids.append(pid)
        else:
            skipped += 1
    print(f"Loaded {len(images)} patients, skipped {skipped} due to no images.")

    # Align texts & metadata only for common patients
    common = list(set(pids) & set(sampled_patient_ids) & set(df['Patient_ID'].tolist()))
    image_dict = {pid: img for pid, img in zip(pids, images) if pid in common}
    label_dict = {pid: lab for pid, lab in zip(pids, labels) if pid in common}
    text_map = {pid: txt for pid, txt in zip(df['Patient_ID'], texts) if pid in common}
    meta_map = {pid: df[df['Patient_ID'] == pid].iloc[0] for pid in common}

    images_filtered = np.array([image_dict[pid] for pid in common])  # (N, 8, H, W, 1)
    labels_filtered = [label_dict[pid] for pid in common]
    texts_filtered = [text_map[pid] for pid in common]
    meta_df_filtered = pd.DataFrame([meta_map[pid] for pid in common])

    # Text features
    vectorizer = TfidfVectorizer(max_features=cfg['tfidf_max_features'])
    text_feats = vectorizer.fit_transform(texts_filtered).toarray()

    # Metadata encoding
    numerical = meta_df_filtered.select_dtypes(include=['float', 'int']).columns.tolist()
    categorical = meta_df_filtered.select_dtypes(include=['object']).drop(columns=['Patient_ID', 'Pathology Classification/ Follow up']).columns.tolist()

    scaler = StandardScaler()
    encoder = OneHotEncoder(sparse_output=False)

    meta_num = scaler.fit_transform(meta_df_filtered[numerical]) if numerical else np.zeros((len(meta_df_filtered), 0))
    meta_cat = encoder.fit_transform(meta_df_filtered[categorical]) if categorical else np.zeros((len(meta_df_filtered), 0))
    meta_feats = np.concatenate([meta_num, meta_cat], axis=1) if (meta_num.size or meta_cat.size) else np.zeros((len(meta_df_filtered), 0))

    # Labels
    labels_encoded, label_names = pd.factorize(labels_filtered)
    labels_cat = to_categorical(labels_encoded)

    # Class weights
    class_weights_vals = compute_class_weight(class_weight='balanced', classes=np.unique(labels_encoded), y=labels_encoded)
    class_weight_dict = dict(enumerate(class_weights_vals))

    # Split
    X_img_temp, X_img_test, X_meta_temp, X_meta_test, X_txt_temp, X_txt_test, y_temp, y_test = train_test_split(
        images_filtered, meta_feats, text_feats, labels_cat, test_size=0.1, random_state=cfg['seed'])

    X_img_train, X_img_val, X_meta_train, X_meta_val, X_txt_train, X_txt_val, y_train, y_val = train_test_split(
        X_img_temp, X_meta_temp, X_txt_temp, y_temp, test_size=0.2, random_state=cfg['seed'])

    return (X_img_train, X_img_val, X_img_test,
            X_meta_train, X_meta_val, X_meta_test,
            X_txt_train, X_txt_val, X_txt_test,
            y_train, y_val, y_test,
            label_names, class_weight_dict)


def extract_image_features(X_img_train, X_img_val, X_img_test, cfg=CONFIG, feat_dim=64):
    feat_paths = [os.path.join(cfg['save_features_dir'], n) for n in cfg['features_names']]
    if all(os.path.exists(p) for p in feat_paths):
        print("Loading saved image features...")
        return [np.load(p) for p in feat_paths]

    # Build encoder (RadImageNet preferred)
    cnn_encoder = create_radimagenet_encoder(cfg, output_dim=feat_dim)

    # Apply TimeDistributed
    seq_input = Input(shape=(cfg['n_views'], cfg['img_size'][0], cfg['img_size'][1], 1))
    td = TimeDistributed(cnn_encoder)(seq_input)
    td_pool = GlobalAveragePooling1D()(td)
    extractor = Model(seq_input, td_pool, name='feature_extractor')

    # Fit small cnn if needed: here we don't train the encoder; it's used for feature extraction.
    X_train_feats = extractor.predict(X_img_train, batch_size=cfg['batch_size'], verbose=1)
    X_val_feats = extractor.predict(X_img_val, batch_size=cfg['batch_size'], verbose=1)
    X_test_feats = extractor.predict(X_img_test, batch_size=cfg['batch_size'], verbose=1)

    # Save
    np.save(feat_paths[0], X_train_feats)
    np.save(feat_paths[1], X_val_feats)
    np.save(feat_paths[2], X_test_feats)

    return X_train_feats, X_val_feats, X_test_feats


def train_and_evaluate(cfg=CONFIG):
    (X_img_train, X_img_val, X_img_test,
     X_meta_train, X_meta_val, X_meta_test,
     X_txt_train, X_txt_val, X_txt_test,
     y_train, y_val, y_test,
     label_names, class_weight_dict) = prepare_dataset(cfg)

    # Extract or load image features
    X_train_feats, X_val_feats, X_test_feats = extract_image_features(X_img_train, X_img_val, X_img_test, cfg)

    # Build Phase2 model
    model = build_multimodal_model(img_feat_dim=X_train_feats.shape[1],
                                   meta_dim=X_meta_train.shape[1] if X_meta_train.size else 0,
                                   text_dim=X_txt_train.shape[1],
                                   n_classes=y_train.shape[1], cfg=cfg)

    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        ModelCheckpoint(cfg['model_ckpt'], monitor='val_loss', save_best_only=True),
        CSVLogger(cfg['metrics_log'])
    ]

    history = model.fit(
        [X_train_feats, X_meta_train, X_txt_train], y_train,
        validation_data=([X_val_feats, X_meta_val, X_txt_val], y_val),
        epochs=cfg['epochs'], batch_size=cfg['batch_size'],
        class_weight=class_weight_dict,
        callbacks=callbacks
    )

    # Evaluate
    preds = model.predict([X_test_feats, X_meta_test, X_txt_test])
    y_pred = np.argmax(preds, axis=1)
    y_true = np.argmax(y_test, axis=1)

    from sklearn.metrics import classification_report, f1_score, accuracy_score, roc_auc_score, confusion_matrix
    print(classification_report(y_true, y_pred, target_names=label_names))
    print("F1 (weighted):", f1_score(y_true, y_pred, average='weighted'))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    try:
        print("ROC AUC:", roc_auc_score(y_test, preds, average='macro', multi_class='ovr'))
    except Exception:
        pass

    return model, history


if __name__ == '__main__':
    # Run training
    train_and_evaluate(CONFIG)
