import tensorflow as tf

# Import necessary modules from TensorFlow and other libraries
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Load the CIFAR-10 dataset, which includes training and testing images and labels
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize the images to have pixel values between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define class names for the CIFAR-10 dataset
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Display the first 25 images from the training set with their labels
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels are arrays, so we need an extra index to get the label
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

# Build the convolutional neural network model
model = models.Sequential()
# Add the first convolutional layer with 32 filters, a 3x3 kernel, and ReLU activation
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# Add a max pooling layer with a 2x2 pool size
model.add(layers.MaxPooling2D((2, 2)))
# Add a second convolutional layer with 64 filters and a 3x3 kernel
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# Add another max pooling layer
model.add(layers.MaxPooling2D((2, 2)))
# Add a third convolutional layer with 64 filters and a 3x3 kernel
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Display the model architecture so far
model.summary()

# Flatten the output from the convolutional layers
model.add(layers.Flatten())
# Add a dense (fully connected) layer with 64 units and ReLU activation
model.add(layers.Dense(64, activation='relu'))
# Add a final dense layer with 10 units (one for each class)
model.add(layers.Dense(10))

# Display the complete model architecture
model.summary()

# Compile the model with the Adam optimizer, sparse categorical crossentropy loss function, and accuracy metric
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model for 10 epochs, using the training data and validating on the test data
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

# Plot the training and validation accuracy over epochs
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

# Evaluate the model on the test data and print the test accuracy
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)
