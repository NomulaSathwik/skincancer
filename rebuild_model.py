import tensorflow as tf
from tensorflow import keras

# Define a new model architecture (Ensure it matches the original)
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')  # Binary classification (Malignant/Benign)
])

# Load extracted weights
weights_file = "models/mymodel-2-weights.h5"
model.load_weights(weights_file)

# Save the rebuilt model in a new format
model.save("models/mymodel-2-converted.h5")
print("✅ Model successfully rebuilt and saved as 'models/mymodel-2-converted.h5'")
