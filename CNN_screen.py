from flask import Flask, request, render_template_string
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

# Caminho do modelo já treinado
MODEL_PATH = "models/tl_vgg16.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Diretório de treino para pegar os nomes das classes
TRAIN_DIR = "dataset/train"
class_names = sorted(os.listdir(TRAIN_DIR))

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            # Pré-processa a imagem
            img = load_img(file, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            img_array = img_array.reshape((1, 224, 224, 3))

            # Faz a previsão
            prediction = model.predict(img_array)
            predicted_class = class_names[prediction.argmax()]

            return f"Classe prevista: {predicted_class}"

    return render_template_string("""
        <h1>Upload de Imagem</h1>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file">
            <input type="submit" value="Enviar">
        </form>
    """)

if __name__ == "__main__":
    app.run(debug=True)
