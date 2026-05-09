
# Section 1: Import Libraries

import os, warnings
warnings.filterwarnings('ignore')

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay

from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

gpus = tf.config.list_physical_devices('GPU')
print(f" TensorFlow  : {tf.__version__}")
print(f" Keras       : {keras.__version__}")
print(f" GPU Devices : {[g.name for g in gpus] if gpus else 'CPU only — consider enabling GPU'}")
print(f" All libraries loaded successfully!")


# Section 2: Dataset Loading & Visualization

IMG_SIZE   = 96
BATCH_SIZE = 64
CHANNELS   = 3

REAL_CLASSES = {2, 3, 4, 5, 6}
FAKE_CLASSES = {0, 1, 7, 8, 9}

print("Loading CIFAR-10 ...")
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = tf.keras.datasets.cifar10.load_data()
CLASS_NAMES = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

def make_binary_labels(y_raw, real_set, fake_set):
    y_flat = y_raw.flatten()
    binary = np.zeros_like(y_flat, dtype=np.int32)
    for idx, label in enumerate(y_flat):
        if label in fake_set:
            binary[idx] = 1
    return binary

y_train = make_binary_labels(y_train_raw, REAL_CLASSES, FAKE_CLASSES)
y_test  = make_binary_labels(y_test_raw,  REAL_CLASSES, FAKE_CLASSES)

print(f"   Total Train : {len(x_train_raw):,}  |  Real: {np.sum(y_train==0):,}  Fake: {np.sum(y_train==1):,}")
print(f"   Total Test  : {len(x_test_raw):,}   |  Real: {np.sum(y_test==0):,}   Fake: {np.sum(y_test==1):,}")
print(f"   Class balance: {np.mean(y_train)*100:.1f}% Fake — perfectly balanced")

fig, axes = plt.subplots(2, 10, figsize=(18, 4))
fig.suptitle('CIFAR-10 Binary Split — Real (Top) vs Fake (Bottom)',
             fontsize=13, fontweight='bold')

real_idx = np.where(y_train == 0)[0][:10]
fake_idx = np.where(y_train == 1)[0][:10]

for i in range(10):
    axes[0, i].imshow(x_train_raw[real_idx[i]])
    axes[0, i].set_title(CLASS_NAMES[y_train_raw[real_idx[i]][0]], fontsize=7, color='#16A34A')
    axes[0, i].axis('off')
    axes[1, i].imshow(x_train_raw[fake_idx[i]])
    axes[1, i].set_title(CLASS_NAMES[y_train_raw[fake_idx[i]][0]], fontsize=7, color='#DC2626')
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()
print("Dataset loaded and visualized.")


# Section 3: Data Preprocessing & Augmentation

import cv2

def resize_batch(images, size):
    return np.array([cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
                     for img in images])

print(f"📐 Resizing images: 32×32 → {IMG_SIZE}×{IMG_SIZE} ...")
x_train_resized = resize_batch(x_train_raw, IMG_SIZE)
x_test_resized  = resize_batch(x_test_raw,  IMG_SIZE)

print("Normalizing to [-1, 1] for MobileNetV2 compatibility ...")
x_train_norm = tf.keras.applications.mobilenet_v2.preprocess_input(
    x_train_resized.astype('float32'))
x_test_norm  = tf.keras.applications.mobilenet_v2.preprocess_input(
    x_test_resized.astype('float32'))

print("Building augmentation pipeline ...")
train_datagen = ImageDataGenerator(
    horizontal_flip    = True,
    rotation_range     = 15,
    zoom_range         = 0.15,
    width_shift_range  = 0.1,
    height_shift_range = 0.1,
    brightness_range   = [0.85, 1.15],
    fill_mode          = 'nearest'
)
train_datagen.fit(x_train_norm)

train_gen = train_datagen.flow(x_train_norm, y_train,
                                batch_size=BATCH_SIZE, shuffle=True, seed=SEED)
test_gen  = ImageDataGenerator().flow(x_test_norm, y_test,
                                       batch_size=BATCH_SIZE, shuffle=False)

print(f"\n Preprocessing complete.")
print(f" Input shape  : {x_train_norm.shape[1:]}  (H × W × C)")
print(f" Train batches: {len(train_gen)}")
print(f" Test  batches: {len(test_gen)}")


# Section 4: Model 1 — CNN Only (MobileNetV2 + Dense Head)

def build_cnn_only(input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS)):
    inputs = keras.Input(shape=input_shape, name='cnn_input')

    base = MobileNetV2(
        input_shape = input_shape,
        include_top = False,
        weights     = 'imagenet',
        input_tensor= inputs
    )
    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D(name='gap')(x)   # (batch, 1280)
    x = layers.Dense(256, activation='relu', name='fc1')(x)
    x = layers.BatchNormalization(name='bn_fc1')(x)
    x = layers.Dropout(0.4, name='drop1')(x)
    output = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = keras.Model(inputs, output, name='CNN_Only')
    model.compile(
        optimizer = Adam(learning_rate=1e-3),
        loss      = 'binary_crossentropy',
        metrics   = ['accuracy',
                     tf.keras.metrics.Precision(name='precision'),
                     tf.keras.metrics.Recall(name='recall')]
    )
    return model

cnn_model = build_cnn_only()
print("CNN-Only Model Summary:")
print("=" * 55)
cnn_model.summary()
trainable   = sum([np.prod(v.shape) for v in cnn_model.trainable_weights])
untrainable = sum([np.prod(v.shape) for v in cnn_model.non_trainable_weights])
print(f"\n   Trainable params   : {trainable:,}")
print(f"   Frozen base params : {untrainable:,}")


# Section 5: Model 2 — CapsNet Only

class SquashActivation(layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, s):
        sq_norm    = tf.reduce_sum(tf.square(s), axis=self.axis, keepdims=True)
        scale      = sq_norm / (1.0 + sq_norm)
        unit_vec   = s / (tf.sqrt(sq_norm) + keras.backend.epsilon())
        return scale * unit_vec

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'axis': self.axis})
        return cfg


class CapsuleBlock(layers.Layer):
    def __init__(self, num_capsules=16, capsule_dim=8, **kwargs):
        super().__init__(**kwargs)
        self.num_capsules = num_capsules
        self.capsule_dim  = capsule_dim
        self.squash       = SquashActivation()

    def build(self, input_shape):
        flat_dim = int(np.prod(input_shape[1:]))
        self.projection = layers.Dense(
            self.num_capsules * self.capsule_dim,
            use_bias=True,
            name='capsule_projection'
        )
        super().build(input_shape)

    def call(self, inputs):
        x = layers.Flatten()(inputs)
        x = self.projection(x)
        x = tf.reshape(x, (-1, self.num_capsules, self.capsule_dim))
        x = self.squash(x)
        x = layers.Flatten()(x)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'num_capsules': self.num_capsules, 'capsule_dim': self.capsule_dim})
        return cfg


def build_capsnet_only(input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS)):
    inputs = keras.Input(shape=input_shape, name='caps_input')

    x = layers.Conv2D(64,  (3,3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(256, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = CapsuleBlock(num_capsules=16, capsule_dim=8, name='capsule_block')(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    output = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = keras.Model(inputs, output, name='CapsNet_Only')
    model.compile(
        optimizer = Adam(learning_rate=1e-3),
        loss      = 'binary_crossentropy',
        metrics   = ['accuracy',
                     tf.keras.metrics.Precision(name='precision'),
                     tf.keras.metrics.Recall(name='recall')]
    )
    return model

caps_model = build_capsnet_only()
print("CapsNet-Only Model Summary:")
print("=" * 55)
caps_model.summary()


# Section 6: Model 3 — Hybrid CNN + CapsNet (Proposed)

def build_hybrid_model(input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS)):
    inputs = keras.Input(shape=input_shape, name='hybrid_input')

    base = MobileNetV2(
        input_shape  = input_shape,
        include_top  = False,
        weights      = 'imagenet',
        input_tensor = inputs
    )
    for layer in base.layers[:-30]:
        layer.trainable = False
    for layer in base.layers[-30:]:
        layer.trainable = True

    x = base.output
    x = CapsuleBlock(num_capsules=16,
                     capsule_dim=8,
                     name='capsule_block')(x)

    x = layers.Dense(256, activation='relu', name='fc1')(x)
    x = layers.BatchNormalization(name='bn_fc1')(x)
    x = layers.Dropout(0.5, name='drop1')(x)
    x = layers.Dense(64,  activation='relu', name='fc2')(x)
    x = layers.Dropout(0.3, name='drop2')(x)
    output = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = keras.Model(inputs, output, name='Hybrid_CNN_CapsNet')

    model.compile(
        optimizer = Adam(learning_rate=5e-4),
        loss      = 'binary_crossentropy',
        metrics   = ['accuracy',
                     tf.keras.metrics.Precision(name='precision'),
                     tf.keras.metrics.Recall(name='recall')]
    )
    return model

hybrid_model = build_hybrid_model()
print("Hybrid CNN + CapsNet Model Summary:")
print("=" * 55)
hybrid_model.summary()
trainable = sum([np.prod(v.shape) for v in hybrid_model.trainable_weights])
print(f"\n   Trainable params (fine-tune) : {trainable:,}")


# Section 7: Training All Three Models

EPOCHS = 25

def get_callbacks(model_name):
    return [
        callbacks.EarlyStopping(
            monitor='val_accuracy', patience=7,
            restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=4, min_lr=1e-7, verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=f'best_{model_name}.weights.h5',
            monitor='val_accuracy', save_best_only=True,
            save_weights_only=True, verbose=0
        )
    ]

histories = {}

# Train Model 1: CNN Only
print("\n" + "="*55)
print("Training Model 1: CNN Only")
print("="*55)
histories['cnn'] = cnn_model.fit(
    train_gen,
    epochs          = EPOCHS,
    validation_data = (x_test_norm, y_test),
    callbacks       = get_callbacks('cnn'),
    verbose         = 1
)

# Train Model 2: CapsNet Only
print("\n" + "="*55)
print("Training Model 2: CapsNet Only")
print("="*55)
histories['caps'] = caps_model.fit(
    train_gen,
    epochs          = EPOCHS,
    validation_data = (x_test_norm, y_test),
    callbacks       = get_callbacks('caps'),
    verbose         = 1
)

# Train Model 3: Hybrid
print("\n" + "="*55)
print("Training Model 3: Hybrid CNN + CapsNet")
print("="*55)
histories['hybrid'] = hybrid_model.fit(
    train_gen,
    epochs          = EPOCHS,
    validation_data = (x_test_norm, y_test),
    callbacks       = get_callbacks('hybrid'),
    verbose         = 1
)

print("\n All three models trained successfully!")


# Section 8: Evaluation & Metrics Comparison

def evaluate_model(model, x_test, y_test, model_name):
    y_prob = model.predict(x_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    acc  = accuracy_score(y_test,  y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test,    y_pred, zero_division=0)
    f1   = f1_score(y_test,        y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1-Score  : {f1*100:.2f}%")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
          target_names=['Real', 'Fake'], digits=4))

    return {'name': model_name, 'acc': acc, 'prec': prec,
            'rec': rec, 'f1': f1, 'cm': cm, 'prob': y_prob}

results = {}
results['cnn']    = evaluate_model(cnn_model,    x_test_norm, y_test, 'CNN Only')
results['caps']   = evaluate_model(caps_model,   x_test_norm, y_test, 'CapsNet Only')
results['hybrid'] = evaluate_model(hybrid_model, x_test_norm, y_test, 'Hybrid CNN+CapsNet')

print("\n" + "="*65)
print("  FINAL COMPARISON TABLE")
print("="*65)
print(f"  {'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-"*65)
for key, r in results.items():
    print(f"  {r['name']:<22} {r['acc']*100:>9.2f}% {r['prec']*100:>9.2f}% "
          f"{r['rec']*100:>9.2f}% {r['f1']*100:>9.2f}%")
print("="*65)


# Section 9: Visualizations & Dashboard

fig = plt.figure(figsize=(20, 16))
fig.suptitle('Hybrid CNN + Capsule Network — Complete Analysis Dashboard',
             fontsize=16, fontweight='bold', y=0.98)

C = {'cnn': '#2563EB', 'caps': '#16A34A', 'hybrid': '#DC2626'}
labels = {'cnn': 'CNN Only', 'caps': 'CapsNet Only', 'hybrid': 'Hybrid'}

# Plot 1: Validation Accuracy Curves
ax1 = fig.add_subplot(3, 3, 1)
for key, h in histories.items():
    ax1.plot([v*100 for v in h.history['val_accuracy']],
             color=C[key], linewidth=2.2, label=labels[key])
ax1.set_title('Validation Accuracy', fontweight='bold')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy (%)')
ax1.legend(); ax1.grid(True, alpha=0.3)

# Plot 2: Validation Loss Curves
ax2 = fig.add_subplot(3, 3, 2)
for key, h in histories.items():
    ax2.plot(h.history['val_loss'],
             color=C[key], linewidth=2.2, label=labels[key])
ax2.set_title('Validation Loss', fontweight='bold')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
ax2.legend(); ax2.grid(True, alpha=0.3)

# Plot 3: Accuracy Comparison Bar
ax3 = fig.add_subplot(3, 3, 3)
model_names = [r['name'] for r in results.values()]
acc_vals    = [r['acc']*100 for r in results.values()]
bar_colors  = list(C.values())
bars = ax3.bar(model_names, acc_vals, color=bar_colors, alpha=0.85,
               edgecolor='white', width=0.5)
ax3.set_title('Accuracy Comparison', fontweight='bold')
ax3.set_ylabel('Accuracy (%)'); ax3.set_ylim([70, 100])
ax3.grid(True, alpha=0.3, axis='y')
for bar in bars:
    ax3.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.2,
             f'{bar.get_height():.1f}%', ha='center', va='bottom',
             fontweight='bold', fontsize=10)

# Plots 4-6: Confusion Matrices for each model
for i, (key, r) in enumerate(results.items()):
    ax = fig.add_subplot(3, 3, 4+i)
    sns.heatmap(r['cm'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real','Fake'], yticklabels=['Real','Fake'],
                linewidths=1, linecolor='white',
                annot_kws={'size':13,'weight':'bold'}, ax=ax)
    ax.set_title(f'Confusion Matrix\n{r["name"]}', fontweight='bold')
    ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')

# Plot 7: All Metrics Comparison
ax7 = fig.add_subplot(3, 3, 7)
metrics_keys = ['acc','prec','rec','f1']
metrics_lbl  = ['Accuracy','Precision','Recall','F1-Score']
x_pos = np.arange(len(metrics_keys))
width = 0.25
for i, (key, r) in enumerate(results.items()):
    vals = [r[k]*100 for k in metrics_keys]
    ax7.bar(x_pos + i*width, vals, width,
            label=labels[key], color=list(C.values())[i], alpha=0.85)
ax7.set_title('All Metrics Comparison', fontweight='bold')
ax7.set_xticks(x_pos + width)
ax7.set_xticklabels(metrics_lbl, rotation=10)
ax7.set_ylabel('Score (%)'); ax7.set_ylim([60, 105])
ax7.legend(); ax7.grid(True, alpha=0.3, axis='y')

# Plot 8: Confidence Distribution (Hybrid)
ax8 = fig.add_subplot(3, 3, 8)
probs = results['hybrid']['prob']
ax8.hist(probs[y_test==0], bins=40, alpha=0.6, color='#16A34A', label='Real', edgecolor='white')
ax8.hist(probs[y_test==1], bins=40, alpha=0.6, color='#DC2626', label='Fake', edgecolor='white')
ax8.axvline(0.5, color='black', linestyle='--', lw=2, label='Threshold')
ax8.set_title('Hybrid — Confidence Distribution', fontweight='bold')
ax8.set_xlabel('P(Fake)'); ax8.set_ylabel('Count')
ax8.legend(); ax8.grid(True, alpha=0.3)

# Plot 9: Training vs Validation Accuracy (all 3 models)
ax9 = fig.add_subplot(3, 3, 9)
for key, h in histories.items():
    ax9.plot([v*100 for v in h.history['accuracy']],
             color=C[key], linewidth=2.2, linestyle='--',
             alpha=0.7, label=f'{labels[key]} (train)')
    ax9.plot([v*100 for v in h.history['val_accuracy']],
             color=C[key], linewidth=2.2, label=f'{labels[key]} (val)')
ax9.set_title('Train vs Val Accuracy (All Models)', fontweight='bold')
ax9.set_xlabel('Epoch'); ax9.set_ylabel('Accuracy (%)')
ax9.legend(fontsize=7); ax9.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('analysis_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
print("Dashboard saved as 'analysis_dashboard.png'")


# Section 10: Single Image Prediction

def predict_image(model, image_input):
    if isinstance(image_input, str):
        from tensorflow.keras.preprocessing.image import load_img, img_to_array
        img = load_img(image_input, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img)
    else:
        img_array = cv2.resize(image_input, (IMG_SIZE, IMG_SIZE)).astype('float32')

    img_norm  = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_batch = np.expand_dims(img_norm, axis=0)

    prob  = float(model.predict(img_batch, verbose=0)[0][0])
    label = 'FAKE' if prob >= 0.5 else 'REAL'
    conf  = prob if prob >= 0.5 else (1 - prob)

    return {'label': label, 'confidence': round(conf*100, 2),
            'probability': round(prob, 4), 'img_array': img_array/255.0}


def display_prediction(result):
    """Render prediction result as a visual card."""
    color = '#DC2626' if result['label'] == 'FAKE' else '#16A34A'
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Deepfake Detection Result', fontsize=13, fontweight='bold')
    axes[0].imshow(np.clip(result['img_array'], 0, 1))
    axes[0].axis('off'); axes[0].set_title('Input Image')
    axes[1].axis('off')
    axes[1].text(0.5, 0.75, f"Prediction: {result['label']}",
                 ha='center', fontsize=18, fontweight='bold',
                 color=color, transform=axes[1].transAxes)
    axes[1].text(0.5, 0.55, f"Confidence: {result['confidence']:.1f}%",
                 ha='center', fontsize=14, transform=axes[1].transAxes)
    axes[1].text(0.5, 0.38, f"Fake Probability: {result['probability']:.4f}",
                 ha='center', fontsize=10, color='gray', transform=axes[1].transAxes)
    axes[1].text(0.5, 0.22, "Threshold: 0.5  |  [0 = Real,  1 = Fake]",
                 ha='center', fontsize=9, color='gray',
                 style='italic', transform=axes[1].transAxes)
    plt.tight_layout(); plt.show()


print("Predicting on 6 random test samples:\n")
idxs = np.random.choice(len(x_test_norm), 6, replace=False)
correct = 0
for idx in idxs:
    r = predict_image(hybrid_model, (x_test_resized[idx]))
    actual = 'REAL' if y_test[idx] == 0 else 'FAKE'
    match  = '✅' if r['label'] == actual else '❌'
    correct += int(r['label'] == actual)
    print(f"  Actual: {actual:<6}  Predicted: {r['label']:<6}  "
          f"Confidence: {r['confidence']:.1f}%  {match}")
print(f"\n  Mini-batch accuracy: {correct}/6")

display_prediction(predict_image(hybrid_model, x_test_resized[idxs[0]]))

print("\n── To predict on YOUR OWN image ──────────────────────────────")
print("# from google.colab import files")
print("# uploaded = files.upload()")
print("# fname = list(uploaded.keys())[0]")
print("# result = predict_image(hybrid_model, fname)")
print("# display_prediction(result)")
