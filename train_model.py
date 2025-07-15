import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load cropped images
data, labels = [], []
categories = ["with_mask", "without_mask"]
for category in categories:
    path = os.path.join("dataset", category)
    for img_name in os.listdir(path):
        img = load_img(os.path.join(path, img_name), target_size=(224, 224))
        img = img_to_array(img) / 255.0
        data.append(img)
        labels.append(0 if category == "with_mask" else 1)

x = np.array(data)
y = np.array(labels)

# Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                         width_shift_range=0.2, height_shift_range=0.2,
                         horizontal_flip=True)

# Build model
base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base.input, outputs=x)

for layer in base.layers:
    layer.trainable = False

model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(aug.flow(x_train, y_train, batch_size=32), validation_data=(x_test, y_test), epochs=7)

model.save("model/mask_detector.h5")
