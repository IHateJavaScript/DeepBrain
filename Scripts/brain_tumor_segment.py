





import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from keras import backend as K
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import argparse
from skimage.transform import resize
from PIL import Image


def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.keras.backend.flatten(y_true)  # 使用 tf.keras.backend.flatten
    y_pred_f = tf.keras.backend.flatten(y_pred)  # 使用 tf.keras.backend.flatten
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


### bce_dice_loss = binary_crossentropy_loss + dice_loss
def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def iou_metric(label, pred):
    return tf.py_func(get_iou_vector, [label, pred>0.5], tf.float64)

def run_segmentation(input_filepath):
    # load the model
    model = load_model('/mnt/DATA/home/cuisj1/projects/DeepBrain/ModelTraining/Brain/segmentation/brain_tumor_segment.h5', custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_loss': dice_loss, 'iou_metric': iou_metric})

    val_image = np.array(Image.open(input_filepath).convert('L'))

    # get value counts
    np.unique(val_image, return_counts=True)


    val_image = np.expand_dims(resize(val_image, (128, 128), mode='constant', preserve_range=True), axis=-1)

    val_image = np.expand_dims(val_image, axis=0)

    THRESHOLD = 0.3
    # use the model to predict the mask
    predicted_mask = (model.predict(val_image)>THRESHOLD)*1

    plt.figure(figsize=(8, 8))

    plt.subplot(1, 2, 1)

    plt.imshow(val_image[0, :, :, 0], cmap='gray')

    plt.title('MRI Image')

    plt.axis('off')

    plt.subplot(1, 2, 2)

    plt.imshow(val_image[0, :, :, 0], cmap='gray')

    plt.imshow(np.ones_like(predicted_mask[0, :, :, 0]) - predicted_mask[0, :, :, 0], alpha=0.2, cmap='Set1')

    plt.title('Predicted Mask')

    plt.axis('off')

    plt.tight_layout()

    plt.savefig('brain_tumor_segmentation.png')

    plt.close()


    # 加载输入图片并转换为灰度
    original_image = np.array(Image.open(input_filepath).convert('L'))
    original_size = original_image.shape  # 保存原始图像尺寸

    # 调整大小以适应模型输入
    resized_image = np.expand_dims(resize(original_image, (128, 128), mode='constant', preserve_range=True), axis=-1)
    resized_image = np.expand_dims(resized_image, axis=0)

    # 设置预测阈值并生成掩膜
    THRESHOLD = 0.3
    predicted_mask = (model.predict(resized_image) > THRESHOLD).astype(np.uint8)

    # 恢复掩膜到原图尺寸
    predicted_mask_resized = resize(predicted_mask[0, :, :, 0], original_size, mode='constant', preserve_range=True)
    predicted_mask_resized = (predicted_mask_resized > 0.5).astype(np.uint8)  # 二值化掩膜

    # 保存掩膜为独立图像
    mask_image = Image.fromarray((predicted_mask_resized * 255).astype(np.uint8))  # 转换为 0-255 范围
    mask_image.save("Mask.jpg")



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Example script using argparse")
    parser.add_argument("--input", type=str, help="Input image path")
    parser.add_argument("--job_id", type=str, help="Job ID")


    args = parser.parse_args()


    input = args.input
    upload_dir = os.path.dirname(input)

    os.chdir(upload_dir)

    run_segmentation(input)





























