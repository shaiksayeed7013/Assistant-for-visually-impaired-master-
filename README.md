# Assistant-for-visually-impaired-master-
# Overview
The "Assistant-for-visually-impaired-master" project aims to assist visually impaired individuals by generating and vocalizing textual descriptions of images. This image captioning system uses deep learning techniques to analyze images and produce meaningful captions, which are then converted to speech.

# Features
Image Feature Extraction: Uses a pre-trained VGG16 model to extract features from images.
Caption Generation: Trains a neural network to generate captions for images using the extracted features.
Text-to-Speech Conversion: Converts the generated captions into speech using pyttsx3 and gTTS.
Dataset
This project uses the Flickr 8k Dataset, which contains 8,000 images each with five different textual descriptions. The dataset is included in the repository.

# Installation
1)Clone the repository:

git clone https://github.com/shaiksayeed701/Assistant-for-visually-impaired-master.git
cd Assistant-for-visually-impaired-master

2)Install the required packages:

pip install -r requirements.txt

# Usage
1. Extract Features from Images
Run extract_features.py to extract features from the images in the dataset using the VGG16 model.

 python extract_features.py
 
2. Train the Image Captioning Model

Run trai_model.py to train the image captioning model using the extracted features and captions.

python trai_model.py

3. Generate Image Descriptions and Voice Output

Run inference.py to generate captions for new images and convert them to speech.

python inference.py

Sample Output

The system will display the image and print the generated caption. It will also play an audio description of the image.


Acknowledgements

Flickr 8k Dataset: Flickr 8k Dataset
VGG16 Model: VGG16 - Very Deep Convolutional Networks for Large-Scale Image Recognition
