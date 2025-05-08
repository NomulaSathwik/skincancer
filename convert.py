import tensorflow as tf
from tensorflow import keras
import h5py

# Open the model file with h5py
model_file = "models/mymodel-2.h5"
output_weights_file = "models/mymodel-2-weights.h5"

with h5py.File(model_file, "r") as f:
    keys = list(f.keys())
    print("Keys in model file:", keys)

    if "model_weights" in f:
        print("✅ Model weights found. Extracting...")
        with h5py.File(output_weights_file, "w") as out_f:
            f.copy("model_weights", out_f)
        print(f"✅ Extracted weights saved to: {output_weights_file}")
    else:
        print("❌ No model weights found! The file might be corrupted.")

