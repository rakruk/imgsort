import os
import pickle
import random
import re
from tqdm import tqdm
import numpy as np
import tensorflow as tf

from image_ops import preprocess_data, TARGET_SIZE


def predict_data(imgpath, model):
    pip_image = tf.keras.preprocessing.image.load_img(
        imgpath, color_mode='rgb', target_size=TARGET_SIZE)
    # preprocess and plotlib only works on array type images
    input_arr = tf.keras.preprocessing.image.img_to_array(pip_image)
    input_arr, _ = preprocess_data(input_arr, None)
    # Convert single image to a batch because .predict() only works on batches
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return pip_image, predictions[0]

# def plot_prediction(pip_image, predictions):
#     import matplotlib.pyplot as plt
#     _ = plt.imshow(pip_image)
#     plt.axis('off')
#     plt.title("{} (confidence: {:.2%})".format(
#         LABELS[predictions[0].argmax()], predictions[0][predictions[0].argmax()]))
#     plt.show()


def predict_images(folder, model, threshold):
    # Get all images paths in the mypath directory
    img_files = [i for i in os.listdir(folder) if os.path.isfile( os.path.join(folder, i)) and re.search("\.(jpg|png|gif)$", i)][:3000]
    random.shuffle(img_files)
    img_pred_path = []
    for i in tqdm(range(len(img_files)), smoothing=0.1): #len(allofem) < 10 and 
        image, pred = predict_data( os.path.join(folder, img_files[i]), model)
        # FIXME: treshold validation is currently in wrong method, move it to move_images
        if pred.max()>threshold: 
            img_pred_path.append((None, pred, img_files[i]))
    print("==== images predicted ====")
    return img_pred_path

def move_images(folder, predictions, labels):
    for file in predictions:
        try:
            os.rename(
                os.path.join(folder , file[2]),
                os.path.join(folder , labels[file[1].argmax()] , file[2])
                )
        except OSError as e:
            print("ERROR: Unable to write  because of error :{}" % e)
    print("==== images moved ====")


def main_sort(unsorted_folder, model_path, threshold, y):
    labels = pickle.load(open(os.path.join(model_path, 'labels.pkl'), "rb") )
    model = tf.keras.models.load_model(model_path)
    predictions = predict_images(unsorted_folder, model, threshold)
    move_images(unsorted_folder, predictions, labels)
