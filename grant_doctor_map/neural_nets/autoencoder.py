from typing import Iterable
from tensorflow import keras
from keras.losses import mean_squared_error


def autoencoder(n_input: int,
                n_bottleneck: int,
                n_layers: Iterable[int]):
    """Create an autoencoder model with separate encoder, decoder, and
    complete model for training

    Args:
        n_input (int): the dimensionality of the input data
        n_bottleneck (int): the dimensionality to reduce to
        n_layers (Iterable[int]): the in-between layers
            Rules: 1. must be descending in size
                   2. no layer may be larger than the input (that's a waste)
                   3. let's have some reasonableness about the step sizes
                   it would weird to say (50, 30, 29, 28, 3)
    """
    inputs = keras.layers.Input(shape=(n_input, ))
    x = inputs
    for layer_size in [n_input] + n_layers:
        x = keras.layers.Dense(layer_size, activation='relu')(x)
    bottleneck = keras.layers.Dense(n_bottleneck)(x)

    dec_inputs = keras.layers.Dense(n_layers[-1], activation='relu')(bottleneck)
    y = dec_inputs
    for layer_size in n_layers[1::-1] + [n_input]:
        y = keras.layers.Dense(layer_size, activation='relu')(y)
    
    encoder_model = keras.models.Model(inputs=inputs, outputs=bottleneck)
    full_model = keras.models.Model(inputs=inputs, outputs=y)
    full_model.compile(loss=mean_squared_error, optimizer='adam')

    return encoder_model, full_model


if __name__ == '__main__':
    import read_wine_data

    df = read_wine_data.read()
    labels = df['quality']
    features = df[[col for col in df.columns if col !='quality']]

    encoder_model, training_model = autoencoder(n_input=11,n_bottleneck=2,
                                                n_layers=[8,6,4])
    
    for col in features:
        features[col] -= features[col].min()
        features[col] /= features[col].max()

    print('BEFORE')
    print(encoder_model.predict(features))

    # An autoencoder is defined as fitting the output data equal to the input data
    training_model.fit(features.values,features.values,
                       epochs=8, batch_size=32, shuffle=True)
    
    print("AFTER")
    print(encoder_model.predict(features))