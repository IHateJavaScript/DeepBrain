
# pre-processing Figshare dataset 


'''

https://www.kaggle.com/code/rawanmaghrbi/brain-tumors-unet-multiple-cnn-vgg19-as-encoder

'''



'''

# start docker image
docker run -it --rm --gpus all --entrypoint /bin/bash  -v /mnt/DATA/home/cuisj1/projects:/mnt/DATA/home/cuisj1/projects google_colab

'''





# get all data in one list
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt


data_dir= '/mnt/DATA/home/cuisj1/projects/DeepBrain/ModelTraining/Brain/data/segmentation/dataset/data'
total_image=3064
datalist=[]
labels=[]

os.chdir("/mnt/DATA/home/cuisj1/projects/DeepBrain/ModelTraining/Brain/segmentation")

for i in range(1,total_image+1):
    
  filename=str(i)+".mat"

  data=h5py.File(os.path.join(data_dir,filename),"r")

    
  datalist.append(data)

  if i%150==0:
    print(filename)
  if i == 3063:
    print("Finished")

for i  in range(total_image):
  label =int(datalist[i]["cjdata"]["label"][()])-1

  labels.append(label)

labels=np.array(labels)


print(labels.shape)

num_to_class = {'0': 'meningioma (0)', 
                '1': 'glioma (1)',
                '2': 'pituitary tumor (2)'}
classes, counts = np.unique(labels,return_counts=True)
plt.bar(classes,counts,tick_label=list(num_to_class.values()))

# save the plot
plt.savefig('label_distribution.png')

plt.show()



print(counts)


##############################################################################################################################


from skimage.transform import resize


images = []
for i in range(total_image):
  im = datalist[i]["cjdata"]["image"][()]
  im = np.expand_dims(resize(im, (128, 128), mode="constant", preserve_range=True), axis=-1)
  images.append(im)

print(len(images))
images=np.array(images)
print(images.shape)



masks = []
for i in range(total_image):
  mask = datalist[i]["cjdata"]["tumorMask"][()]
    # 归一化到 [0, 255]
  im_normalized = (im - np.min(im)) / (np.max(im) - np.min(im)) * 255.0

  # 转换数据类型为 uint8
  im_normalized = im_normalized.astype(np.uint8)
  mask = np.expand_dims(resize(mask, (128, 128), mode="constant", preserve_range=True), axis=-1)
  masks.append(mask)

print(len(masks))
masks=np.array(masks)
print(masks.shape)



images = []
for i in range(total_image):
    # 加载原始图像
    im = datalist[i]["cjdata"]["image"][()]
    
    # 归一化到 [0, 255]
    im_normalized = (im - np.min(im)) / (np.max(im) - np.min(im)) * 255.0
    
    # 转换数据类型为 uint8
    im_normalized = im_normalized.astype(np.uint8)
    
    # 调整大小并添加通道维度
    im_resized = resize(im_normalized, (128, 128), mode="constant", preserve_range=True)
    im_resized = np.expand_dims(im_resized, axis=-1)
    
    # 添加到列表
    images.append(im_resized)

# 转为 NumPy 数组
images = np.array(images)

# 输出信息
print(len(images))
print(images.shape)


##############################################################################################################################


from sklearn.model_selection import train_test_split ,RandomizedSearchCV , GridSearchCV 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

input_shape = images[0].shape # input shape
input_shape
x_train, x_test, y_train, y_test= train_test_split(images,masks,test_size=0.2, shuffle= True)
x_train.shape
train_datagen = ImageDataGenerator(brightness_range=(0.9,1.1),
                                   zoom_range=[.9,1.1],
                                   fill_mode='nearest')
val_datagen = ImageDataGenerator()
#augment
x_train= np.append( x_train, [ np.fliplr(x) for x in  x_train], axis=0 )
y_train = np.append( y_train, [ np.fliplr(y) for y in  y_train], axis=0 )
x_train.shape, y_train.shape



from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_vgg19_unet(input_shape):
    """ Input """
    inputs = Input(input_shape)

    # Expand input to 3 channels
    if input_shape[-1] == 1:
        expanded_input = Concatenate()([inputs, inputs, inputs])  # Concatenate along channel dimension
    else:
        expanded_input = inputs

    """ Pre-trained VGG19 Model """
    vgg19 = VGG19(include_top=False, weights="imagenet", input_tensor=expanded_input)

    """ Encoder """
    s1 = vgg19.get_layer("block1_conv2").output         ## (512 x 512)
    s2 = vgg19.get_layer("block2_conv2").output         ## (256 x 256)
    s3 = vgg19.get_layer("block3_conv4").output         ## (128 x 128)
    s4 = vgg19.get_layer("block4_conv4").output         ## (64 x 64)

    """ Bridge """
    b1 = vgg19.get_layer("block5_conv4").output         ## (32 x 32)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)                     ## (64 x 64)
    d2 = decoder_block(d1, s3, 256)                     ## (128 x 128)
    d3 = decoder_block(d2, s2, 128)                     ## (256 x 256)
    d4 = decoder_block(d3, s1, 64)                      ## (512 x 512)

    """ Output """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="VGG19_U-Net")
    return model

# Example usage:
input_shape = (128, 128, 1)  # Input shape with 1 channel
model = build_vgg19_unet(input_shape)
model.summary()



##############################################################################################################################


import tensorflow as tf

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.keras.backend.flatten(y_true)  # 使用 tf.keras.backend.flatten
    y_pred_f = tf.keras.backend.flatten(y_pred)  # 使用 tf.keras.backend.flatten
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


### bce_dice_loss = binary_crossentropy_loss + dice_loss
def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def get_iou_vector(A, B):
    t = A>0
    p = B>0
    intersection = np.logical_and(t,p)
    union = np.logical_or(t,p)
    iou = (np.sum(intersection) + 1e-10 )/ (np.sum(union) + 1e-10)
    return iou

def iou_metric(label, pred):
    return tf.py_func(get_iou_vector, [label, pred>0.5], tf.float64)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
             loss=bce_dice_loss, metrics=['accuracy'])




from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from keras import backend as K


model_checkpoint  = ModelCheckpoint('model_best_checkpoint.keras', save_best_only=True,
                                    monitor='val_loss', mode='min', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
reduceLR = ReduceLROnPlateau(patience=4, verbose=2, monitor='val_loss',min_lr=1e-4, mode='min')

callback_list = [early_stopping, reduceLR, model_checkpoint]

# train_generator = train_datagen.flow(x_train, y_train, batch_size=32)
# val_generator = val_datagen.flow(x_test, y_test, batch_size=32)

hist = model.fit(x_train,y_train, validation_data=(x_test,y_test),epochs=5,batch_size=4)




model.save('brain_tumor_segment.h5')


##############################################################################################################################

# load the model

import tensorflow as tf

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


# load the model
model = load_model('/mnt/DATA/home/cuisj1/projects/DeepBrain/ModelTraining/Brain/segmentation/brain_tumor_segment.h5', custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_loss': dice_loss, 'iou_metric': iou_metric})

# model = load_model('segment.h5')



##############################################################################################################################



THRESHOLD = 0.3
predicted_mask = (model.predict(x_test)>THRESHOLD)*1

plt.figure(figsize=(8,30))
i=1;total=10
temp = np.ones_like(y_test[0] )
for idx in np.random.randint(0,high=x_test.shape[0],size=total):
    plt.subplot(total,3,i);i+=1
    plt.imshow( x_test[idx], cmap='gray' )
    plt.title("MRI Image");plt.axis('off')

    plt.subplot(total,3,i);i+=1
    plt.imshow( x_test[idx], cmap='gray' )
    plt.imshow( temp - y_test[idx], alpha=0.2, cmap='Set1' )
    plt.title("Original Mask");plt.axis('off')

    plt.subplot(total,3,i);i+=1
    plt.imshow( x_test[idx], cmap='gray' )
    plt.imshow( temp - predicted_mask[idx],  alpha=0.2, cmap='Set1' )
    plt.title("Predicted Mask");plt.axis('off')

plt.tight_layout()

# save the plot
plt.savefig('segmentation_results.png')
plt.show()






plot_model(model, to_file='model_segnet.png', show_shapes=True, show_layer_names=True)




os.chdir("/mnt/project/DeepBrain/Results/Segmentation/Validate")

val_filepath = '/mnt/project/DeepBrain/Data/ashkhagan/figshare-brain-tumor-dataset/versions/1/dataset/data/12.mat'

val_data = h5py.File(val_filepath, 'r')

val_image= val_data['cjdata']['image'][()]


# get value counts
np.unique(val_image, return_counts=True)

# 归一化到 [0, 255]
im_normalized = (im - np.min(val_image)) / (np.max(val_image) - np.min(val_image)) * 255.0

# 转换数据类型为 uint8
im_normalized = im_normalized.astype(np.uint8)


val_image = np.expand_dims(resize(val_image, (128, 128), mode='constant', preserve_range=True), axis=-1)

val_mask = val_data['cjdata']['tumorMask'][()]

val_mask = np.expand_dims(resize(val_mask, (128, 128), mode='constant', preserve_range=True), axis=-1)

val_image = np.expand_dims(val_image, axis=0)

val_mask = np.expand_dims(val_mask, axis=0)

val_image.shape, val_mask.shape

THRESHOLD = 0.3
# use the model to predict the mask
predicted_mask = (model.predict(val_image)>THRESHOLD)*1

plt.figure(figsize=(8, 8))

plt.subplot(1, 3, 1)

plt.imshow(val_image[0, :, :, 0], cmap='gray')

plt.title('MRI Image')

plt.axis('off')

plt.subplot(1, 3, 2)

plt.imshow(val_image[0, :, :, 0], cmap='gray')

plt.imshow(np.ones_like(val_mask[0, :, :, 0]) - val_mask[0, :, :, 0], alpha=0.2, cmap='Set1')

plt.title('Original Mask')

plt.axis('off')

plt.subplot(1, 3, 3)

plt.imshow(val_image[0, :, :, 0], cmap='gray')

plt.imshow(np.ones_like(predicted_mask[0, :, :, 0]) - predicted_mask[0, :, :, 0], alpha=0.2, cmap='Set1')

plt.title('Predicted Mask')

plt.axis('off')

plt.tight_layout()

plt.savefig('val_segmentation_results.png')






