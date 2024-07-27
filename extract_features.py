import os
import pickle
from tqdm import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

BASE_DIR = 'Flickr 8k Dataset'
features = {}
directory = os.path.join(BASE_DIR, 'Images')

# Load VGG16 model and modify it
base_model = VGG16()
model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)

for i in tqdm(os.listdir(directory)):
    img_path = os.path.join(directory, i)
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    image_id = i.split('.')[0]
    features[image_id] = feature

with open(os.path.join(BASE_DIR, 'features.pkl'), 'wb') as f:
    pickle.dump(features, f)
