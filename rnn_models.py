import keras

def _finishBuildingRNN(model, optimizer, loss):
    model.summary()
    model.compile(optimizer=optimizer, loss=loss)
    return model

# Trying out a network largely based on:
# Wang, Juxing, and Shen, Linyong. "Semi-Adaptable Human Hand Motion Prediction 
# Based on Neural Networks and Kalman Filter." Journal of Physics: Conference 
# Series. Vol. 2029. No. 1. IOP Publishing, 2021.
def get_fcnn_rnn(loss, win_size: int, vec_dims: int, out_size: int = 3,
                 optimizer = None):
    if optimizer is None:
        optimizer = keras.optimizers.Adam(0.0001)

    model = keras.Sequential([
        keras.layers.Input((win_size, vec_dims)),
        keras.layers.Flatten(),
        keras.layers.Dense(12, activation='relu'),
        keras.layers.Dense(20, activation='relu'),
        # keras.layers.Dense(18, activation='relu'),
        keras.layers.Dense(out_size)#num_classes, activation='sigmoid')
    ])

    return _finishBuildingRNN(model, optimizer, loss)



def get_simple_lstm(loss, out_size=3, optimizer='adam',
                    units=(32, 32), dropout=0.1, noise=0.0):
    layers = []  # win_size may be None
    if noise > 0:
        layers.append(keras.layers.GaussianNoise(noise))  # only active in training
    for i, u in enumerate(units):
        last = (i == len(units) - 1)
        layers.append(keras.layers.LSTM(u, return_sequences=not last))
        if dropout > 0:
            layers.append(keras.layers.Dropout(dropout))
    layers.append(keras.layers.Dense(out_size))
    
    return _finishBuildingRNN(keras.Sequential(layers), optimizer, loss)

def get_frame_context_lstm(loss, win_size, ctx_dims, units=(50, 50), optimizer='adam'):
    seq_in = keras.layers.Input((win_size, 3), name="local_window")
    ctx_in = keras.layers.Input((ctx_dims,), name="global_ctx")

    x = seq_in
    for i, u in enumerate(units):
        x = keras.layers.LSTM(u, return_sequences=(i < len(units) - 1))(x)

    c = keras.layers.Dense(16, activation='relu')(ctx_in)   # small, so it can't dominate
    h = keras.layers.Concatenate()([x, c])
    h = keras.layers.Dense(32, activation='relu')(h)         # lets motion and context interact
    out = keras.layers.Dense(3)(h)

    model = keras.Model([seq_in, ctx_in], out)
    return _finishBuildingRNN(model, optimizer, loss)

