





import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
#---------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
#---------------------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#---------------------------------------
import warnings
warnings.filterwarnings("ignore")

import argparse





def predict(img_path, job_id, model, output_path):
    
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    import json
    
    label = list(['glioma', 'meningioma', 'notumor', 'pituitary'])
    plt.figure(figsize=(12, 12))
    img = Image.open(img_path)
    resized_img = img.resize((299, 299))
    img = np.asarray(resized_img)
    img = np.expand_dims(img, axis=0)
    img = img / 255
    predictions = model.predict(img)
    probs = list(predictions[0])
    labels = label
    plt.subplot(2, 1, 1)
    plt.imshow(resized_img)
    plt.subplot(2, 1, 2)
    bars = plt.barh(labels, probs)
    plt.xlabel('Probability', fontsize=15)
    ax = plt.gca()
    ax.bar_label(bars, fmt = '%.2f')
    
    os.chdir(output_path)
    plt.savefig(f'Prediction.png')
    plt.show()
    
    class_type = "normal"
    
    if label[np.argmax(predictions)] == 'normal':
        class_type = 'normal'
    else:
        class_type = 'tumor'

    # save the result to json, image_name,for class, get the highest probability, for label, get the label name
    result = {'image_name': img_path.split('/')[-1], 'class': class_type, 'label': label[np.argmax(predictions)]}
    print(result)
    # write the result to a json file
    with open(f'brain_tumor_detect.json', 'w') as f:
        json.dump(result, f)



# main function
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Example script using argparse")
    parser.add_argument("--input", type=str, help="Input image path")
    parser.add_argument("--job_id", type=str, help="Job ID")

    args = parser.parse_args()

    
    # load the model
    saved_model = tf.keras.models.load_model('/mnt/DATA/home/cuisj1/projects/DeepBrain/ModelTraining/Brain/detection/Brain_Tumor_Detection_Model.h5')

    # get the path of the input image
    upload_dir = os.path.dirname(args.input)

    predict(img_path = args.input, 
            model = saved_model,
            job_id = args.job_id,
            output_path = upload_dir)





























