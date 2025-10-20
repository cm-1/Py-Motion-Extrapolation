import keras

def _finishBuildingRNN(model, optimizer, loss):
    model.summary()
    model.compile(optimizer=optimizer, loss=loss)
    return model

# Trying out a network largely based on:
# Wang, Juxing, and Shen, Linyong. "Semi-Adaptable Human Hand Motion Prediction 
# Based on Neural Networks and Kalman Filter." Journal of Physics: Conference 
# Series. Vol. 2029. No. 1. IOP Publishing, 2021.
def get_fcnn_rnn(win_size: int, vec_dims: int, out_size: int = 3,
                 optimizer = None, loss = 'mse'):
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



def get_simple_lstm(out_size = 3, optimizer = 'adam', loss='mse'):
    model = keras.Sequential([
        # keras.layers.LSTM(128, return_sequences=True),
        keras.layers.LSTM(50),
        keras.layers.Dense(out_size) #num_classes, activation='sigmoid')
    ])

    return _finishBuildingRNN(model, optimizer, loss)



