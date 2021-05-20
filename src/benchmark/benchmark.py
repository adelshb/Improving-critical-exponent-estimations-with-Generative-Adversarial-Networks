import matplotlib.pyplot as plt
import tensorflow as tf
import ast

from src.statphy.models.percolation import generate_data

model = tf.keras.models.load_model("/Users/matthieu.sarkis/git_repos/Improving-critical-exponent-estimations-with-Generative-Adversarial-Networks/saved_models/CNN_L128_N10000/saved-model.h5")

X, _, _ = generate_data(L=128, 
                        p_arr=[0.5928],
                        max_configs_per_p=1000)

file = open("/Users/matthieu.sarkis/git_repos/Improving-critical-exponent-estimations-with-Generative-Adversarial-Networks/saved_models/CNN_L128_N10000/labels.json", "r")
contents = file.read().strip()
temp = ast.literal_eval(contents)
file.close()
reversed_labels = {value : float(key) for (key, value) in temp.items()}

y_pred = model.predict(X).argmax(axis=1)
y_pred = [reversed_labels[i] for i in y_pred]

plt.hist(y_pred)
plt.title("Distribution of the value of p for GAN generated critical configurations")
plt.savefig('saved_files/hist_GANgenerated_configs.png')
