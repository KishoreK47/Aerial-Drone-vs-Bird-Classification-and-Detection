
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from pathlib import Path

def get_dataloaders(data_dir, img_size=(224,224), batch_size=32):
    
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.7,1.3]
    )

    test_val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)
    
    train_gen = train_datagen.flow_from_directory(
        Path(data_dir) / "train",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary"
    )

    val_gen = test_val_datagen.flow_from_directory(
        Path(data_dir) / "valid",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary"
    )

    test_gen = test_val_datagen.flow_from_directory(
        Path(data_dir) / "test",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False
    )

    return train_gen, val_gen, test_gen
