import os
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from gtts import gTTS
import pyttsx3
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

BASE_DIR = 'Flickr 8k Dataset'

# Load pre-extracted features
with open(os.path.join(BASE_DIR, 'features.pkl'), 'rb') as f:
    features = pickle.load(f)

# Load the trained model
model = load_model('best_model.h5')

# Load the tokenizer
with open(os.path.join(BASE_DIR, 'tokenizer.pkl'), 'rb') as f:
    tokenizer = pickle.load(f)

# Load the mapping dictionary
with open(os.path.join(BASE_DIR, 'captions.txt'), 'r') as f:
    next(f)
    desc_doc = f.read()

mapping = {}
for each_desc in desc_doc.split('\n'):
    tokens = each_desc.split(',')
    if len(each_desc) < 2:
        continue
    image_id, desc_of = tokens[0], tokens[1:]
    image_id = image_id.split('.')[0]
    desc_of = " ".join(desc_of)
    if image_id not in mapping:
        mapping[image_id] = []
    mapping[image_id].append(desc_of)

max_length = 35  # Set this to the max_length used in your training

def mapping_toword(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_description(model, image, tokenizer, max_length):
    in_text = 'beginning'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        desc_predict = model.predict([image, sequence], verbose=0)
        desc_predict = np.argmax(desc_predict)
        word = mapping_toword(desc_predict, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'ending':
            break
    return in_text

def voice_output(command):
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()

def generate_text(image_name):
    image_id = image_name.split('.')[0]
    img_path = os.path.join(BASE_DIR, "Images", image_name)
    image = Image.open(img_path)
    desc = mapping[image_id]  # Load the descriptions for the image
    y_pred = predict_description(model, features[image_id], tokenizer, max_length)
    plt.imshow(image)
    plt.show()
    text = y_pred.split(' ', 1)[1].rsplit(' ', 1)[0]
    tts = gTTS(text)
    tts.save('info.wav')
    voice_output(text)
    return text

# Example usage
image_name = "111766423_4522d36e56.jpg"  # Change to the image name you want to test
text = generate_text(image_name)
print(text)

image_name = "109202801_c6381eef15.jpg"  # Change to the image name you want to test
text = generate_text(image_name)
print(text)
