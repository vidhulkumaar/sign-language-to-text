import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_path = r"D:\sign language\dataSet1\testingData"
img_size = (32, 32)
batch_size = 32

# Use the same data generator settings as in training
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Only need to load once to get class_indices
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Save the class label mapping
with open("class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f)

print("✅ class_indices.json created successfully.")
