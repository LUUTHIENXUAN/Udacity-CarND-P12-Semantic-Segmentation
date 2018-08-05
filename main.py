import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import os.path
import tensorflow as tf
slim = tf.contrib.slim

import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import random
import time
import math
import numpy as np
import matplotlib.pyplot as plt

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

all_training_losses = []

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name      = 'image_input:0'
    vgg_keep_prob_tensor_name  = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph       = tf.get_default_graph()

    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob   = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out  = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out  = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out  = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    # display operation
    print ('[load_vgg]Load the model and weights')
    for op in graph.get_operations():
        print('', op.name)
    # view
    # tf.summary.FileWriter('graph_log', graph = graph)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.
    Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    print("\nIn layers...")
    regularizer = tf.contrib.layers.l2_regularizer(1e-3)

    init        = tf.truncated_normal_initializer(stddev = 0.01)
    init_ran    = tf.random_normal_initializer(stddev=0.01)

    # scale down sd
    # https://github.com/MarvinTeichmann/KittiSeg
    sd = 0.01
    init_var_f  = tf.contrib.layers.variance_scaling_initializer()
    init_var_s  = tf.contrib.layers.variance_scaling_initializer(factor=2.0*sd)
    init_var_t  = tf.contrib.layers.variance_scaling_initializer(factor=2.0*sd*sd)

    # Encoder
    def conv_1x1(layer, num_classes,
                 init = init, regularizer = regularizer):

        return tf.layers.conv2d(inputs  = layer,
                                filters = num_classes,
                                kernel_size = (1, 1),
                                strides = (1, 1),
                                padding = 'same',
                                kernel_initializer=init,
                                kernel_regularizer = regularizer)

    # Decoder
    def upsample(layer, num_classes, k, s,
                 init = init, regularizer = regularizer):
        return tf.layers.conv2d_transpose(inputs  = layer,
                                          filters = num_classes,
                                          kernel_size = (k, k),
                                          strides = (s, s),
                                          padding = 'same',
                                          kernel_initializer=init,
                                          kernel_regularizer = regularizer)

    # Encoder
    layer7_conv_1x1 = conv_1x1(layer       = vgg_layer7_out,
                               num_classes = num_classes,
                               init        = init_var_f)
    layer4_conv_1x1 = conv_1x1(layer       = vgg_layer4_out,
                               num_classes = num_classes,
                               init        = init_var_s)
    layer3_conv_1x1 = conv_1x1(layer       = vgg_layer3_out,
                               num_classes = num_classes,
                               init        = init_var_t)


    # Decoder
    layer7_output   = upsample(layer       = layer7_conv_1x1,
                               num_classes = num_classes,
                               k =5, s = 2) # 5->4 failed

    print(' Shape of layer7 = ',
          tf.Print(layer7_output, [tf.shape(layer7_output)[1:3]]))
    layer7_output   = tf.layers.batch_normalization(layer7_output)
    # Skip Connection
    layer7_skip     = tf.add(layer7_output , layer4_conv_1x1)

    # Decoder
    layer4_output   = upsample(layer       = layer7_skip,
                               num_classes = num_classes,
                               k = 5, s = 2) # 5->4 failed
    print(' Shape of layer4 = ',
          tf.Print(layer4_output, [tf.shape(layer4_output)[1:3]]))
    layer4_output   = tf.layers.batch_normalization(layer4_output)

    # Skip Connection
    layer4_skip     = tf.add(layer4_output, layer3_conv_1x1)

    # Decoder
    layer3_output   = upsample(layer       = layer4_skip,
                               num_classes = num_classes,
                               k = 14, s = 8) # 14->16 failed
    #tf.Print(output, [tf.shape(output)[1:3]])
    print(' Shape of layer3 = ',
          tf.Print(layer3_output, [tf.shape(layer3_output)[1:3]]))

    return layer3_output
tests.test_layers(layers)
helper.graph_visualize()

def optimize(nn_last_layer, correct_label, learning_rate,
             num_classes, method = "RMSP"):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    print("\nIn optimize...")


    logits             = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label      = tf.reshape(correct_label, (-1, num_classes))

    if (method == "Adam"):
        # Regularization loss collector
        # https://stackoverflow.com/questions/37107223/how-to-add-regularizations-in-tensorflow
        reg_losses         = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_constant       = 0.01  # Choose an appropriate one.

        # normal loss
        correct_label      = tf.reshape(correct_label, (-1, num_classes))
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = correct_label))

        cross_entropy_loss = cross_entropy_loss + reg_constant* sum(reg_losses)
        train_op           = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)


    if (method == "RMSP"):
        weights = [0.3, 0.6]

        softmax            =  tf.nn.softmax(logits) + tf.constant(value= 0.000000001)
        cross_entropy      = -tf.reduce_sum(tf.multiply(correct_label * tf.log(softmax), weights),
                                            reduction_indices=[1])
        cross_entropy_loss =  tf.reduce_mean(cross_entropy, name='xentropy_mean')

        reg_loss_col       = tf.GraphKeys.REGULARIZATION_LOSSES
        weight_loss        = sum(tf.get_collection(reg_loss_col))
        cross_entropy_loss = cross_entropy_loss + weight_loss * 0.0000001

        train_op           =  tf.train.RMSPropOptimizer(learning_rate = learning_rate,
                                                        decay = 0.9,
                                                        epsilon =0.000000001).minimize(cross_entropy_loss)

    if (method == "Focal_loss"):
        """
        [REF] github.com/unsky/focal-loss
        """

        keep_inds     = np.where( correct_label != -1)[0]
        correct_label = correct_label[keep_inds]

        cls   = logits[keep_inds, correct_label]
        cls  += 1e-14

        gamma    = 2
        alpha    = 0.25
        cls_loss = alpha * (-1.0 * np.power(1- cls, gamma) * np.log(cls))
        cls_loss = np.sum(cls_loss)/len(correct_label)

        cross_entropy_loss = cls_loss

        train_op =  tf.train.RMSPropOptimizer(learning_rate = learning_rate,
                                              decay = 0.9,
                                              epsilon =0.000000001).minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)

def train_nn(sess, epochs, batch_size,
             get_batches_fn, train_op, cross_entropy_loss,
             input_image, correct_label, keep_prob,
             learning_rate, augment_image):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.
           Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    print ("\nIn train_nn:")

    # Create queue coordinator.
    coord   = tf.train.Coordinator()
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # image normalization
    #_R_MEAN = 123.68
    #_G_MEAN = 116.78
    #_B_MEAN = 103.94

    # Visulization
    plot_losses, plot_samples = [], []
    sample = 0

    for epoch in range(epochs):

        losses, i = [], 0
        for image, label in get_batches_fn(batch_size):

            start_time = time.time()
            i += 1

            # mean subtraction normalization
            #image = image - [_R_MEAN, _G_MEAN, _B_MEAN]

            # Augmenotation
            #image, label = augment_image(image, label)

            # Training
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict = {input_image: image,
                                            correct_label:label,
                                            keep_prob:0.75,
                                            learning_rate:0.0001})

            losses.append(loss)
            # Visulization
            plot_losses.append(loss)
            plot_samples.append(sample)

            duration = time.time() - start_time
            print("---> iteration: ", i,
                  "Training Loss: {:.4f}...({:.3f} sec/step)".format(loss, duration))

            sample = sample + batch_size

        training_loss = sum(losses) / len(losses)
        all_training_losses.append(training_loss)

        print("------------------")
        print("epoch: ", epoch + 1, " of ", epochs, "training loss: ", training_loss)
        print("------------------")


    plt.plot(plot_samples, plot_losses, 'k-', label= "Train Loss")
    #plt.plot(plot_samples, all_training_losses,'r--', label = "Average Loss")
    plt.title("Training Loss")
    plt.xlabel("Sample")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')
    plt.savefig('runs/E%04d-B%04d-K%f.png'%(epochs, batch_size, 0.75))

    coord.request_stop()
    coord.join(threads)
tests.test_train_nn(train_nn)

def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset
    #           instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    epochs     = 50
    batch_size = 8

    # Allocate fixed memory
    #https://stackoverflow.com/questions/34199233/
    #how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
    config = tf.ConfigProto()
    # only allocate 90% of the total memory of each GPU
    config.gpu_options.allocator_type ='BFC'
    config.gpu_options.per_process_gpu_memory_fraction = 0.90

    print("\nIn run...")
    with tf.Session(config=config) as sess:

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        augment_image = helper.image_augmentation()

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)

        layer_output  = layers(layer3_out, layer4_out, layer7_out, num_classes)

        correct_label = tf.placeholder(tf.float32, shape = [None, None, None, num_classes])

        learning_rate = tf.placeholder(tf.float32)

        # TODO: Train NN using the train_nn function
        logits, train_op, cross_entropy_loss = optimize(layer_output, correct_label,
                                                        learning_rate, num_classes)

        sess.run(tf.global_variables_initializer())

        train_nn(sess, epochs, batch_size, get_batches_fn, train_op,
                 cross_entropy_loss, input_image, correct_label, keep_prob,
                 learning_rate, augment_image)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape,
                                      logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video

if __name__ == '__main__':
    run()
