# -------------------------------
# PARTE 1 – PRÉ-PROCESSAMENTO
# -------------------------------

import os, random, shutil
from glob import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 224, 224, 3
SEED = 42
BASE_DIR = "images"
PREP_DIR = "prepared"
TRAIN_DIR, VAL_DIR, TEST_DIR = [os.path.join(PREP_DIR, split) for split in ["train","val","test"]]

def ensure_dir(path): os.makedirs(path, exist_ok=True)
for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]: ensure_dir(d)

classes = sorted([d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))])
print("Classes detectadas:", classes)

def preprocess_image(path):
    img = Image.open(path).convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT))
    arr = np.array(img).astype(np.float32)/255.0
    return arr, img

def save_preprocessed_image(img_pil, output_path):
    ensure_dir(os.path.dirname(output_path))
    img_pil.save(output_path, format="PNG")

data = [(f, cls) for cls in classes for f in glob(os.path.join(BASE_DIR, cls, "*.png"))]
random.Random(SEED).shuffle(data)
paths, labels = zip(*data)

train_paths, test_paths, train_labels, test_labels = train_test_split(paths, labels, test_size=0.2, stratify=labels, random_state=SEED)
train_paths, val_paths, train_labels, val_labels = train_test_split(train_paths, train_labels, test_size=0.2, stratify=train_labels, random_state=SEED)

def process_and_save(dataset_paths, dataset_labels, split_dir):
    for src, cls in zip(dataset_paths, dataset_labels):
        _, img_pil = preprocess_image(src)
        dst = os.path.join(split_dir, cls, os.path.basename(src))
        save_preprocessed_image(img_pil, dst)

process_and_save(train_paths, train_labels, TRAIN_DIR)
process_and_save(val_paths, val_labels, VAL_DIR)
process_and_save(test_paths, test_labels, TEST_DIR)

# -------------------------------
# PARTE 2A – CNN DO ZERO
# -------------------------------

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import itertools

BATCH_SIZE, EPOCHS, LR = 8, 10, 1e-3

train_gen = ImageDataGenerator(rescale=1./255, rotation_range=10,
                               width_shift_range=0.05, height_shift_range=0.05,
                               zoom_range=0.1, horizontal_flip=True).flow_from_directory(
    TRAIN_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True, seed=SEED)

val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    VAL_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    TEST_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

num_classes = train_gen.num_classes

def build_simple_cnn(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), num_classes=num_classes):
    model = models.Sequential([
        layers.Conv2D(32,(3,3),activation='relu',padding='same',input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,(3,3),activation='relu',padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128,(3,3),activation='relu',padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128,activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes,activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

simple_cnn = build_simple_cnn()
simple_cnn.summary()
history_simple = simple_cnn.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)
test_loss, test_acc = simple_cnn.evaluate(test_gen)
print("Acurácia CNN do zero:", test_acc)

y_pred = np.argmax(simple_cnn.predict(test_gen), axis=1)
print(classification_report(test_gen.classes, y_pred, target_names=list(test_gen.class_indices.keys())))

cm = confusion_matrix(test_gen.classes, y_pred)
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(6,6))
    cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão")
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45)
    plt.yticks(ticks, classes)
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i,f"{cm[i,j]:.2f}",ha="center",color="white" if cm[i,j]>cm.max()/2 else "black")
    plt.ylabel("Verdadeiro")
    plt.xlabel("Previsto")
    plt.show()
plot_confusion_matrix(cm, list(test_gen.class_indices.keys()))

# -------------------------------
# PARTE 2B – TRANSFER LEARNING
# -------------------------------

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

tl_train_gen = ImageDataGenerator(preprocessing_function=preprocess_input, rotation_range=10,
                                  width_shift_range=0.05, height_shift_range=0.05,
                                  zoom_range=0.1, horizontal_flip=True).flow_from_directory(
    TRAIN_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True)

tl_val_gen = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    VAL_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

tl_test_gen = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    TEST_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
for layer in base_model.layers: layer.trainable = False

x = layers.Flatten()(base_model.output)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(tl_train_gen.num_classes, activation='softmax')(x)
tl_model = models.Model(inputs=base_model.input, outputs=outputs)
tl_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
tl_model.summary()

history_tl = tl_model.fit(tl_train_gen, validation_data=tl_val_gen, epochs=8)
for layer in base_model.layers[-4:]: layer.trainable = True
tl_model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
history_tl_ft = tl_model.fit(tl_train_gen, validation_data=tl_val_gen, epochs=4)

tl_test_loss, tl_test_acc = tl_model.evaluate(tl_test_gen)
print("Acurácia Transfer Learning:", tl_test_acc)

tl_y_pred = np.argmax(tl_model.predict(tl_test_gen), axis=1)
print(classification_report(tl_test_gen.classes, tl_y_pred, target_names=list(tl_test_gen.class_indices.keys())))
tl_cm = confusion_matrix(tl_test_gen.classes, tl_y_pred)
plot_confusion_matrix(tl_cm, list(tl_test_gen.class_indices.keys()))

# -------------------------------
# PARTE 3 – PROTÓTIPO FLASK
# -------------------------------

from flask import Flask, request, render_template_string
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)
MODEL_PATH = "models/tl_vgg16.h5"
tl_model.save(MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
class_names = sorted(os.listdir(TRAIN_DIR))

TEMPLATE = """
<!doctype html>
<title>Assistente Cardiológico Virtual</title>
<h2>Classificação de Imagens Médicas</h2>
<form method=post enctype=multipart/form-data>
  <input type=file name=file>
  <input type=submit value="Classificar">
</form>
{% if probs %}
  <h3>Resultado</h3>
  <ul>
  {% for cls, p in probs %}
    <li>{{ cls }}: {{ "%.3f"|format(p) }}</li>
  {% endfor %}
  </ul>
{% endif %}
"""