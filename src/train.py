print("🚀 Script started")

import tensorflow as tf
print("✅ TensorFlow imported")

from model import build_cnn
print("✅ Model imported")

# Load dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
print("✅ Dataset loaded")

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

print("✅ Data preprocessed")

# Build model
model = build_cnn()
print("✅ Model built")

# Train
model.fit(X_train, y_train, epochs=3, validation_data=(X_test, y_test))

# Save
model.save("model/digit_model.h5")

print("🎉 Training completed")