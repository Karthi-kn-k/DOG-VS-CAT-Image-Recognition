import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from google.colab import files
import shutil

# Step 1: Dataset path
# data_path = "/kaggle/input/dog-and-cat-classification-dataset/PetImages" --> this is used in colab for using the data set by import it from kaggle
# IN VS CODE 
data_path = "PetImages"


# Step 2: Prepare working directory (because original files may contain corrupt images)
working_dir = "/tmp/cats_and_dogs"
os.makedirs(working_dir, exist_ok=True)
shutil.copytree(data_path, working_dir, dirs_exist_ok=True)

# Step 3: Remove corrupted images
def clean_corrupted_images(folder):
    for category in ['Cat', 'Dog']:
        folder_path = os.path.join(folder, category)
        for img_file in os.listdir(folder_path):
            try:
                img_path = os.path.join(folder_path, img_file)
                img = tf.keras.utils.load_img(img_path)
            except Exception:
                os.remove(img_path)

clean_corrupted_images(working_dir)

# Step 4: Data generators
img_size = (128, 128)
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    working_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    working_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Step 5: Build the model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # binary output
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Step 6: Train the model
model.fit(train_gen, epochs=5, validation_data=val_gen)

# Step 7: Upload and predict
uploaded = files.upload()

for fname in uploaded.keys():
    img = load_img(fname, target_size=img_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = "🐶 Dog" if prediction > 0.5 else "🐱 Cat"
    print(f"{fname} → {label}")