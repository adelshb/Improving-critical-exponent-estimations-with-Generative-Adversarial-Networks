import matplotlib.pyplot as plt
import tensorflow as tf
import ast
import json

from src.statphy.models.percolation import generate_data

model = tf.keras.models.load_model("saved_models/CNN_L128_N10000/saved-model.h5")

X, _, _ = generate_data(L=128, 
                        p_arr=[0.5928],
                        max_configs_per_p=1000)

with open("saved_models/CNN_L128_N10000/labels.json", 'r') as f:
    labels = json.load(f)

reversed_labels = {value : float(key) for (key, value) in labels.items()}

y_pred = model.predict(X).argmax(axis=1)
y_pred = [reversed_labels[i] for i in y_pred]

plt.hist(y_pred)
plt.title("Distribution of the value of p for GAN generated critical configurations")
plt.savefig('saved_files/hist_GANgenerated_configs.png')
