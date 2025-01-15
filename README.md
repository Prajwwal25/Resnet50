# Metal Surface Defect Detection using ResNet50

## Overview
This project implements a deep learning solution for detecting and classifying metal surface defects using a fine-tuned ResNet50 model. The system is designed to identify 8 different types of surface defects in metal materials, making it useful for quality control in manufacturing environments.

### Dataset Statistics
- **Total Images**: 1,800 grayscale images
- **Image Format**: Grayscale, BMP format
- **Image Resolution**: 200 × 200 pixels
- **Classes**: 6 distinct defect types
- **Images per Class**: 300 samples per defect type
- **File Size**: Total size approximately 1.2 GB

### Defect Types
1. **Rolled-in Scale (RS)**
   - Appears as dark elongated regions
   - Caused by rolled-in oxide scale during rolling process
   - 300 images

2. **Patches (Pa)**
   - Appears as lighter regions with irregular shapes
   - Results from uneven surface oxidation
   - 300 images

3. **Crazing (Cr)**
   - Network of fine lines or cracks on the surface
   - Caused by thermal or mechanical stress
   - 300 images

4. **Pitted Surface (PS)**
   - Small pits or cavities on the metal surface
   - Results from localized corrosion or manufacturing defects
   - 300 images

5. **Inclusion (In)**
   - Foreign particles embedded in the metal surface
   - Usually appears as dark spots
   - 300 images

6. **Scratches (Sc)**
   - Linear marks or grooves on the surface
   - Mechanical damage during handling or processing
   - 300 images

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

### Data Split
- **Training Set**: 80% (1,440 images)
- **Test Set**: 20% (360 images)
- **Stratified Split**: Equal distribution of defect types in both sets

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


### Source
- Dataset available on Kaggle: [NEU Metal Surface Defects Dataset](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)
- Original source: [NEU Database](http://faculty.neu.edu.cn/songkc/en/zhym/263024/list/index.htm)











