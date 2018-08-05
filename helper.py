import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
from tensorflow.python.platform import gfile

from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        """
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)
        """

        # Download vgg
        print('Downloading pre-trained vgg model...')
        """
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)
        """
        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))

def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        background_color = np.array([255, 0, 0])

        # image normalization
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images    = []
            gt_images = []

            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image    = scipy.misc.imresize(scipy.misc.imread(image_file),
                                               image_shape)
                # mean subtraction normalization
                image = image - [_R_MEAN, _G_MEAN, _B_MEAN]

                # random adding effects
                image    = color_distortion(image)

                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file),
                                               image_shape)

                gt_bg    = np.all(gt_image == background_color, axis=2)
                gt_bg    = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                images.append(image)
                gt_images.append(gt_image)

            # random mirroring
            images,gt_images = image_mirroring(np.array(images), np.array(gt_images))

            yield images,gt_images
            #yield np.array(images), np.array(gt_images)

    return get_batches_fn

def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):

    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep probability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """

    # image normalization
    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94

    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):

        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        # mean subtraction normalization
        image = image - [_R_MEAN, _G_MEAN, _B_MEAN]

        im_softmax = sess.run([tf.nn.softmax(logits)], {keep_prob: 1.0, image_pl: [image]})

        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])

        # Accept all pixel with conf >= 0.5 as positive prediction
        # threshold = 0.5 -> 0.6
        segmentation = (im_softmax > 0.6).reshape(image_shape[0], image_shape[1], 1)
        mask         = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask         = scipy.misc.toimage(mask, mode="RGBA")

        street_im    = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)

def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits,
                           keep_prob, input_image):

    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(sess, logits, keep_prob, input_image,
                                    os.path.join(data_dir, 'data_road/testing'),
                                    image_shape)

    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)

######################################################
####################### CRF ##########################
# https://github.com/Gurupradeep/FCN-for-Semantic-Segmentation/blob/master/CRF.ipynb
# http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/18/
# image-segmentation-with-tensorflow-using-cnns-and-conditional-random-fields/


######################################################
############## Visulization ##########################
def graph_visualize():

    # Path to vgg model
    data_dir = './data'
    vgg_path = os.path.join(data_dir, 'vgg')

    with tf.Session() as sess:

        model_filename = os.path.join(vgg_path, 'saved_model.pb')

        with gfile.FastGFile(model_filename, 'rb') as f:

            data = compat.as_bytes(f.read())
            sm   = saved_model_pb2.SavedModel()

            sm.ParseFromString(data)

            g_in = tf.import_graph_def(sm.meta_graphs[0].graph_def)

    LOGDIR = '.'
    train_writer = tf.summary.FileWriter(LOGDIR)
    train_writer.add_graph(sess.graph)
    train_writer.flush()
    train_writer.close()

######################################################
############## Augmentation ##########################
def color_distortion(img):

    # INPUT: single image
    def add_brightness(img):
        return tf.image.random_brightness(img, max_delta = 32. / 255.)

    # INPUT: single image
    def add_saturation(img):
        return tf.image.random_saturation(img, lower = 0.5, upper = 1.5)

    # INPUT: single image
    def add_hue(img):
        return tf.image.random_hue(img, max_delta = 0.2)

    # INPUT: single image
    def add_contrast(img):
        return tf.image.random_contrast(img, lower = 0.5, upper = 1.5)

    #https://medium.com/@lisulimowicz/tensorflow-cpus-and-gpus-configuration-9c223436d4ef
    order = random.randint(0,3)

    if (order == 0):

        with tf.device('/cpu:0'):
            img = add_brightness(img)
            img = add_saturation(img)
            img = add_hue(img)
            img = add_contrast(img)

            # convert tensor object to Numpy array # tf.Session().run(img)
            img = img.eval()

    elif (order == 1):

        with tf.device('/cpu:0'):
            img = add_saturation(img)
            img = add_hue(img)
            img = add_contrast(img)
            img = add_brightness(img)

            # convert tensor object to Numpy array # tf.Session().run(img)
            img = img.eval()

    elif (order == 2):

        with tf.device('/cpu:0'):
            img = add_hue(img)
            img = add_contrast(img)
            img = add_brightness(img)
            img = add_saturation(img)

            # convert tensor object to Numpy array # tf.Session().run(img)
            img = img.eval()

    elif (order == 3):

        with tf.device('/cpu:0'):
            img = add_contrast(img)
            img = add_brightness(img)
            img = add_saturation(img)
            img = add_hue(img)

            # convert tensor object to Numpy array # tf.Session().run(img)
            img = img.eval()

    else:
        raise ValueError('order out of range [0,1]')

    return img

def image_mirroring(img, label):
    """
    Randomly mirrors the images.
    [REF:] github.com/DrSleep/tensorflow-deeplab-resnet
    """
    with tf.device('/cpu:0'):
        distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
        mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
        mirror = tf.boolean_mask([0, 1, 2], mirror)

        img   = tf.reverse(img, mirror)
        label = tf.reverse(label, mirror)

        img   = img.eval()
        label = label.eval()

    return img, label

def random_modification(img, label):
    """
    [REF:] github.com/DrSleep/tensorflow-deeplab-resnet
    1. def image_scaling(img, label)
    2. def image_mirroring(img, label)
    3. random_crop_and_pad_image_and_labels(image, label, ignore_label=255)
    """

    def image_scaling(img, label):
        """
        Randomly scales the images between 0.5 to 1.5 times the original size.
        """

        scale     = tf.random_uniform([1], minval=0.5, maxval=1.5, dtype=tf.float32, seed=None)
        h_new     = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
        w_new     = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))
        new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])

        img   = tf.image.resize_images(img, new_shape)
        label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
        label = tf.squeeze(label, squeeze_dims=[0])

        return img, label

    def image_mirroring(img, label):
        """
        Randomly mirrors the images.
        """

        distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
        mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
        mirror = tf.boolean_mask([0, 1, 2], mirror)

        img   = tf.reverse(img, mirror)
        label = tf.reverse(label, mirror)

        return img, label

    def random_crop_and_pad_image_and_labels(img, label, ignore_label=255):
        """
        Randomly crop and pads the input images.
        """

        label    = tf.cast(label, dtype=tf.float32)
        label    = label - ignore_label # Needs to be subtracted and later added due to 0 padding.
        combined = tf.concat(axis=2, values=[img, label])

        image_shape  = tf.shape(img)
        crop_h       = image_shape[0] - random.uniform(0,1) *image_shape[0]
        crop_w       = image_shape[1] - random.uniform(0,1) *image_shape[1]

        #crop_h       = tf.cast (crop_h, dtype=tf.float32)
        #crop_w       = tf.cast (crop_w, dtype=tf.float32)


        combined_pad = tf.image.pad_to_bounding_box(combined,
                                                    0,
                                                    0,
                                                    tf.maximum(crop_h, image_shape[0]),
                                                    tf.maximum(crop_w, image_shape[1]))

        last_image_dim = tf.shape(img)[-1]
        last_label_dim = tf.shape(label)[-1]
        combined_crop  = tf.random_crop(combined_pad, [crop_h,crop_w,4])

        img_crop   = combined_crop[:, :, :last_image_dim]

        label_crop = combined_crop[:, :, last_image_dim:]
        label_crop = label_crop + ignore_label
        label_crop = tf.cast(label_crop, dtype=tf.uint8)

        # Set static shape so that tensorflow knows shape at compile time.
        img_crop.set_shape((crop_h, crop_w, 3))
        label_crop.set_shape((crop_h,crop_w, 1))

        return img_crop, label_crop

    order = random.randint(0,1)

    if (order == 0):
        with tf.device('/cpu:0'):
            #img, label = image_scaling(img, label)
            img, label = img, label

    elif (order == 1):
        with tf.device('/cpu:0'):
            img, label = image_mirroring(img, label)

    else:
        raise ValueError('order out of range [0,1]', order)

    #img   = img.eval()
    #label = label.eval()

    return img, label

def image_augmentation():

    def conbri_img(img):
        s   = random.uniform(0.85, 1.25) # Contrast augmentation
        m   = random.randint(-35, 35)    # Brightness augmentation

        img = img.astype(np.int)
        img = img * s + m
        img[img > 255] = 255
        img[img < 0] = 0
        img = img.astype(np.uint8)
        return img

    # INPUT: 4D image batch [batch_size, width, height, channels]
    def color_distortion(imgs):

        # INPUT: single image
        def add_brightness(img):
            return tf.image.random_brightness(img, max_delta = 32. / 255.)

        # INPUT: single image
        def add_saturation(img):
            return tf.image.random_saturation(img, lower = 0.5, upper = 1.5)

        # INPUT: single image
        def add_hue(img):
            return tf.image.random_hue(img, max_delta = 0.2)

        # INPUT: single image
        def add_contrast(img):
            return tf.image.random_contrast(img, lower = 0.5, upper = 1.5)


        order = random.randint(0,3)

        if (order == 0):
            # tf.map_fn applies the single-image operator to each element of the batch
            img = tf.map_fn(lambda img: add_brightness(img), imgs)
            img = tf.map_fn(lambda img: add_saturation(img), imgs)
            img = tf.map_fn(lambda img: add_hue(img), imgs)
            img = tf.map_fn(lambda img: add_contrast(img), imgs)

            # convert tensor object to Numpy array # tf.Session().run(img)
            img = img.eval()

        elif (order == 1):
            # tf.map_fn applies the single-image operator to each element of the batch
            img = tf.map_fn(lambda img: add_saturation(img), imgs)
            img = tf.map_fn(lambda img: add_hue(img), imgs)
            img = tf.map_fn(lambda img: add_contrast(img), imgs)
            img = tf.map_fn(lambda img: add_brightness(img), imgs)

            # convert tensor object to Numpy array # tf.Session().run(img)
            img = img.eval()

        elif (order == 2):
            # tf.map_fn applies the single-image operator to each element of the batch
            img = tf.map_fn(lambda img: add_hue(img), imgs)
            img = tf.map_fn(lambda img: add_contrast(img), imgs)
            img = tf.map_fn(lambda img: add_brightness(img), imgs)
            img = tf.map_fn(lambda img: add_saturation(img), imgs)

            # convert tensor object to Numpy array # tf.Session().run(img)
            img = img.eval()

        elif (order == 3):
            # tf.map_fn applies the single-image operator to each element of the batch
            img = tf.map_fn(lambda img: add_contrast(img), imgs)
            img = tf.map_fn(lambda img: add_brightness(img), imgs)
            img = tf.map_fn(lambda img: add_saturation(img), imgs)
            img = tf.map_fn(lambda img: add_hue(img), imgs)

            # convert tensor object to Numpy array # tf.Session().run(img)
            img = img.eval()

        else:
            raise ValueError('order out of range [0,3]')

        return img

    def image_mirroring(img, label):
        """
        Randomly mirrors the images.
        [REF]...github.com/DrSleep/tensorflow-deeplab-resnet/..
        """
        distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
        mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
        mirror = tf.boolean_mask([0, 1, 2], mirror)

        img   = tf.reverse(img, mirror)
        label = tf.reverse(label, mirror)

        # convert tensor object to Numpy array # tf.Session().run(img)
        img   = img.eval()
        label = label.eval()

        return img, label

    def augment(img, label):

        # randomly add some color
        img = color_distortion(img)

        with tf.device('/cpu:0'):
            # randomly modify it size
            img, label = image_mirroring(img, label)

        return img, label

    return augment
