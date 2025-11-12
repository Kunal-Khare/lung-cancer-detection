Medical Image Classification (CNN)
Overview

This project uses a Convolutional Neural Network (CNN) to classify medical images as Normal or Abnormal (cancer).
Images are resized to 150×150 pixels and trained to help identify possible cancer indicators.

Features

📂 Loads and preprocesses images from train/ and test/ folders

🧠 CNN built with TensorFlow/Keras

📊 Visualizes dataset and training progress

🔍 Predicts whether an image is Normal or Abnormal

💬 Displays health suggestions based on predictions

Requirements

Install the dependencies:

pip install numpy scikit-learn seaborn matplotlib opencv-python tensorflow tqdm pandas

Folder Structure
project/
├── train/
│   ├── normal/
│   └── abnormal/
├── test/
│   ├── normal/
│   └── abnormal/


Each folder contains corresponding medical images (e.g., X-rays, scans).

Model Architecture

A simple CNN built with Keras:

2 × Conv2D + MaxPooling layers

Flatten + Dense layers

Output layer for 2 classes (Normal/Abnormal)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])


Compiled and trained using:

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

Visualizations

Bar charts for class distribution

Sample image grids

Pie chart of class ratios

pd.DataFrame({'train': train_counts, 'test': test_counts}, index=class_names).plot.bar()

Running the Project

Prepare dataset as shown above

Run the script to train and test the CNN

Check predictions — the model prints and visualizes whether an image is Normal or Abnormal

python main.py

Example Output

Prediction: Abnormal

Suggestion: “Possible cancer detected — consult a doctor immediately.”

Next Steps

Add data augmentation for better generalization

Use precision, recall, F1-score for deeper analysis

Try transfer learning (e.g., VGG16, ResNet) for improved accuracy
