from tensorflow import keras 
from tensorflow.keras import layers 

model = keras.Sequential([
    #hidden l1
    layers.Dense(units=4, activation='relu', input_shape = [2]),
    #hidden l2
    layers.Dense(units=3, activation = 'relu'),
    #linear output 
    layers.Dense(units=1)
])