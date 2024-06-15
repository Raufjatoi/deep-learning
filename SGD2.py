# Import necessary libraries
import matplotlib.pyplot as plt
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split

# Load the dataset
fuel = pd.read_csv('fuel.csv')

# Separate features and target
X = fuel.copy()
y = X.pop('FE')

# Preprocessing pipelines for numeric and categorical features
preprocessor = make_column_transformer(
    (StandardScaler(), make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(sparse_output=False), make_column_selector(dtype_include=object)),
)

# Apply preprocessing to the features
X = preprocessor.fit_transform(X)

# Log transform the target variable
y = np.log(y)

# Define the neural network model
from tensorflow import keras
from tensorflow.keras import layers

input_shape = [X.shape[1]]  # Set input shape based on preprocessed X

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(128, activation='relu'),    
    layers.Dense(64, activation='relu'),
    layers.Dense(1),
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="mae",
)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    verbose=1
)

# Plot the training history
history_df = pd.DataFrame(history.history)
# Start the plot at epoch 5
history_df.loc[5:, ['loss']].plot()
plt.title('Model Loss After Epoch 5')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
