import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical 
import matplotlib.pyplot as plt

# Load and preprocess data
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape images to include channel dimension
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)) 

# One-hot encode labels (use different variable names)
train_labels_cat = to_categorical(train_labels)
test_labels_cat = to_categorical(test_labels)

# Build the CNN model
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels_cat, epochs=5, batch_size=64, 
          validation_data=(test_images, test_labels_cat))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels_cat)
print(f"Test accuracy: {test_acc*100:.2f}%")

# Make predictions
predictions = model.predict(test_images)
print(f"Predicted label for first test image: {np.argmax(predictions[0])}")

# Display an example image with its prediction
plt.imshow(test_images[0].reshape(28, 28), cmap='gray')
plt.title(f"Predicted label: {(predictions[0].argmax())}")
plt.show()