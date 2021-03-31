from tensorflow import keras

def create_model(L, K):

    # input layer
    i = keras.layers.Input(shape=(L,L,1))

    # Convolution block
    x = keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(i)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2,2))(x)
    
    x = keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2,2))(x)
    
    x = keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2,2))(x)

    # Classification block
    x = keras.layers.Flatten()(x)
    #x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    #x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    #x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(K, activation='softmax')(x)

    model = keras.models.Model(i, x)
    return model
###########

    
    