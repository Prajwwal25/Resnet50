# Metal Surface Defect Detection using ResNet50

## Overview
This project implements a deep learning solution for detecting and classifying metal surface defects using a fine-tuned ResNet50 model. The system is designed to identify 8 different types of surface defects in metal materials, making it useful for quality control in manufacturing environments.

## Features
- Pre-trained ResNet50 architecture with custom modifications
- Advanced data augmentation for improved model generalization
- Fine-tuning capabilities for transfer learning
- Real-time prediction visualization
- Training history monitoring and visualization
- Automated learning rate adjustment
- Early stopping to prevent overfitting

## Installation

### Hardware Requirements
```
- Minimum 8GB RAM
- GPU support recommended for faster training
```

### Software Dependencies
```python
tensorflow >= 2.0.0
numpy
matplotlib
PIL
```

### Dataset Structure
```
NEU Metal Surface Defects Data/
├── train/
│   ├── Crazing/
│   ├── Patches/
│   ├── Rolled/
│   └── [Other defect types]/
└── test/
    ├── Crazing/
    ├── Patches/
    ├── Rolled/
    └── [Other defect types]/
```

## Implementation

### Model Architecture
```python
model = Sequential([
    ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3)),
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.4),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(8, activation='softmax')
])
```

### Training Configuration
```python
optimizer = Adam(learning_rate=0.0001)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### Data Augmentation
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
```

## Usage

### Training the Model
```python
# Initialize model
model = create_improved_model()

# Prepare data
train_generator, validation_generator, test_generator = create_data_generators(
    train_dir='path/to/train',
    test_dir='path/to/test'
)

# Train model
history = train_model(model, train_generator, validation_generator)
```

### Making Predictions
```python
# Load and preprocess image
image = tf.keras.preprocessing.image.load_img(
    image_path, 
    target_size=(128, 128)
)
input_array = tf.keras.preprocessing.image.img_to_array(image)
input_array = np.expand_dims(input_array, axis=0) / 255.0

# Make prediction
prediction = model.predict(input_array)
```

## Model Performance

### Training Metrics
```
Training accuracy: ~85-90%
Validation accuracy: ~80-85%
Test accuracy: ~80%
```

### Training Parameters
```
Batch size: 32
Initial learning rate: 0.0001
Maximum epochs: 10
Early stopping patience: 5
Learning rate reduction factor: 0.2











