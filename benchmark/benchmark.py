import tensorflow as tf
from statphy.models.percolation import generate_data

model = tf.keras.models.load_model('../saved_models/CNN_L128_N10000/saved-models.h5')

X, _, _ = generate_data(L=128, 
                        p_arr=[0.5928],
                        max_configs_per_p=10)

y_pred = model.predict(X)

