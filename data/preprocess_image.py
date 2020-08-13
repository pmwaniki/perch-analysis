import tensorflow as tf

import  matplotlib.pyplot as plt
import numpy as np
import pydicom
from PIL import Image


#load image from disk and convert to array
def image2array(full_path,shape=(224,224,3)):
    h,w,d=shape
    grayscale=d==1
    # raise Exception("Stopped for no reason")
    if type(full_path)==bytes:  full_path=full_path.decode('utf-8')
    # print(full_path)
    if '.dcm' in str(full_path):
        im = pydicom.read_file(full_path)
        image_array = im.pixel_array
        image_array = image_array / image_array.max()
        image_array = (255 * image_array).clip(0, 255)  # .astype(np.int32)
        image_array = Image.fromarray(image_array)
    else:
        image_array = Image.open(str(full_path))
        #image_array = image_array.convert("RGB")

    if grayscale:
        image_array=image_array.convert("L")
    else:
        image_array=image_array.convert("RGB")
    image_array = image_array.resize((h,w))
    image_array = np.asarray(image_array)/255.0
    if grayscale:
        image_array=image_array.reshape((h,w,1))
    return image_array








def preprocess(image,shape,scale=True):
  image = tf.image.decode_jpeg(image, channels=shape[2])
  image = tf.image.resize(image, shape[:2])
  if scale: image = image /255.0
  return image

def preprocess2(file,shape,scale=True):
  image = tf.io.read_file(file)
  image=tf.cond(
      tf.strings.regex_full_match(file,".+png$"),
      lambda: tf.image.decode_png(image,channels=shape[2]),
      lambda: tf.image.decode_jpeg(image,channels=shape[2])
  )
  image = tf.image.resize(image, shape[:2])
  if scale: image /= 255.0  # normalize to [0,1] range
  return image


def load_and_preprocess_image(file,shape,scale=True):
    image = tf.io.read_file(file)
    image2=preprocess(image,shape,scale=scale)
    return image2

def load_and_preprocess_image_ensemble(im,rev,shape,scale=True):
    im2=load_and_preprocess_image(im,shape=shape,scale=scale)
    return im2,rev

def load_and_preprocess_multitask(path_perch,lab_perch,path_chestray,lab_chestray,shape,scale=True):
    im_perch = tf.io.read_file(path_perch)
    im_perch2=preprocess(im_perch,shape,scale=scale)
    im_chestray = tf.io.read_file(path_chestray)
    im_chestray2 = preprocess(im_chestray, shape,scale=scale)
    return im_perch2,lab_perch,im_chestray2,lab_chestray



def load_and_preprocess_multitask_ensemble(path_perch,lab_perch,rev,path_chestray,lab_chestray,shape,scale=True):
    im_perch2, lab_perch, im_chestray2, lab_chestray=\
        load_and_preprocess_multitask(path_perch,lab_perch,path_chestray,lab_chestray,shape=shape,scale=scale)
    return im_perch2, lab_perch,rev, im_chestray2, lab_chestray


def binarize(image,threshold=0.5):
    if threshold is None:
        threshold=tf.random.uniform([],minval=0.4,maxval=0.6,dtype=tf.float32)
    return tf.cast(image>threshold,tf.float32)

#IMAGE AUGUMENTATION






def plot_images(dataset, n_images, samples_per_image,shape=(224,224,3),transform=None):
    height,width,depth=shape
    output = np.zeros((height * n_images, width * samples_per_image, depth))

    row = 0
    for images in dataset.repeat(samples_per_image).batch(n_images):
        output[:, row*width:(row+1)*height] = np.vstack(images.numpy())
        row += 1

    if transform is not None:
        output=transform(output)

    plt.figure(figsize=(12.0,12.0))
    if depth==1:
        plt.imshow(output[:,:,0],cmap="gray",vmin=0,vmax=1)
        plt.show()
    else:
        plt.imshow(output)
        plt.show()


def subtract(x,value=0.5):
    return x-value

def divide(x,value=255.0):
    return x/value

def multiply(x,value=2.0):
    return x*value


def flip(x):
    x=tf.image.random_flip_left_right(x)
    x=tf.image.random_flip_up_down(x)
    return x


def normalize(x):
    return tf.image.per_image_standardization(x)


def color(x,cont_lower=0.3,cont_upper=0.9,bright_delta=0.1):
    x=tf.image.random_brightness(x,bright_delta)
    x=tf.image.random_contrast(x,cont_lower,cont_upper)
    return x

def rotate(x: tf.Tensor):
    # Rotate 0, 90, 180, 270 degrees
    return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))


def crop_and_pad(x,proportion=0.20,width=0.8,height=0.8):
    im_width,im_height,im_channels=x.shape
    crop_width=tf.cast(tf.floor(im_width*width),tf.int32)
    crop_height=tf.cast(tf.floor(im_height*height),tf.int32)
    def crop(x):
        x=tf.image.random_crop(x,(crop_width,crop_height,im_channels))
        ofset_y=tf.random.uniform([],minval=0,maxval=im_height-crop_height,dtype=tf.int32)
        ofset_x = tf.random.uniform([],minval=0,maxval=im_width-crop_width,dtype=tf.int32)
        x = tf.image.pad_to_bounding_box(x,offset_height=ofset_y,offset_width=ofset_x, target_width=im_width, target_height=im_height)
        return x
    x=tf.cond(tf.random.uniform([], 0, 1) < proportion,lambda :crop(x),lambda:x)
    return x

def crop_and_resize(x,width=0.8,height=0.8,proportion=0.8):
    def crop(x):
        im_width, im_height, im_channels = x.shape
        crop_width = tf.cast(tf.floor(im_width * width), tf.int32)
        crop_height = tf.cast(tf.floor(im_height * height), tf.int32)
        x = tf.image.random_crop(x, (crop_width, crop_height, im_channels))
        x = tf.image.resize(x, size=(im_height, im_width))
        return x
    x=tf.cond(tf.random.uniform([], 0, 1) < proportion,lambda :crop(x),lambda:x)

    return x


def augument_multitask(im_perch,lab_perch,im_chestray,lab_chestray,aug_list=[]):
    for aug in aug_list:
        im_perch=aug(im_perch)
        im_chestray=aug(im_chestray)
    return im_perch,lab_perch,im_chestray,lab_chestray

def augument_multitask_ensemble(im_perch,lab_perch,rev,im_chestray,lab_chestray,aug_list=[]):
    for aug in aug_list:
        im_perch=aug(im_perch)
        im_chestray=aug(im_chestray)
    return im_perch,lab_perch,rev,im_chestray,lab_chestray

def augment_ensemble(im,rev,aug_funs=[]):
    for fun in aug_funs:
        im=fun(im)
    return im,rev
