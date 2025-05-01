import pandas as pd
import numpy as np
from keras.models import Sequential

# Load training data
train_df = pd.read_csv('train_with_images_p.csv')
X_train = train_df['entity_name']
y_train = train_df.iloc[1:].values

print(train_df.info())

print(X_train.shape)
print(y_train.shape)

# Load test data
test_df = pd.read_csv('test_images_p.csv')
X_test = test_df.values  # Assuming test data doesn't have labels

# Reshape data to fit CNN input requirements (e.g., 28x28 images with 1 channel)
# This is an example, adjust dimensions according to your data
img_height, img_width = 28, 28
X_train = X_train.reshape(-1, img_height, img_width, 1)
X_test = X_test.reshape(-1, img_height, img_width, 1)

# Normalize data
X_train = X_train / 255.0
X_test = X_test / 255.0


# Encode labels
num_classes = len(np.unique(y_train))
y_train = to_categorical(y_train, num_classes=num_classes)

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_height, img_width, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Predict on the test data
predictions = model.predict(X_test)

# Convert predictions to class labels
predicted_classes = np.argmax(predictions, axis=1)

# Save predictions to a CSV file
output_df = pd.DataFrame({'Prediction': predicted_classes})
output_df.to_csv('predictions.csv', index=False)
