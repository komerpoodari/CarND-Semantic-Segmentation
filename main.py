#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


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
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    # added logic begins - 1
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    # print (graph)
    image_input = graph.get_tensor_by_name (vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name (vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name (vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name (vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name (vgg_layer7_out_tensor_name)
    return (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    # added logic ends - 1
    
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function #komer 2
    # added logic begins - 2

    
    #define regularizer
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1) #fine tuning was needed
    # 1x1 convolution on VGG Layer 7 output
    layer7_c1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same', 
                                   kernel_initializer= tf.random_normal_initializer(stddev=0.01), #fine-tuning needed
                                   kernel_regularizer= regularizer)

    # then upsample 
    layer7_ups = tf.layers.conv2d_transpose(layer7_c1x1, num_classes, 4, strides=(2,2), padding='same', 
                                             kernel_initializer= tf.random_normal_initializer(stddev=0.01), #fine-tuning needed
                                             kernel_regularizer= regularizer)

    # 1x1 convolve Layer 4 output 
    layer4_c1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same', 
                                   kernel_initializer= tf.random_normal_initializer(stddev=0.01), #fine-tuning needed
                                   kernel_regularizer= regularizer)

    #  Add layer7_ups and layer4_c1x1  as inputs to layer8_out
    layer74_out =  tf.add(layer7_ups, layer4_c1x1)
    
    # Upsample Layer 8 out
    layer374_input1 = tf.layers.conv2d_transpose(layer74_out, num_classes, 4,  strides= (2, 2), padding= 'same', 
                                             kernel_initializer= tf.random_normal_initializer(stddev=0.01), #fine-tuning needed
                                             kernel_regularizer= regularizer)
                                             
    # 1x1 convolve layer3 output
    layer374_input2 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same', 
                                   kernel_initializer= tf.random_normal_initializer(stddev=0.01), #fine-tuning needed
                                   kernel_regularizer= regularizer)
                              
    # Add layer374 input1 and input2
    layer374_out = tf.add(layer374_input1, layer374_input2)
    
    # Upsample by 8
    last_layer = tf.layers.conv2d_transpose(layer374_out, num_classes, 16,  strides= (8, 8), padding= 'same', 
                                             kernel_initializer= tf.random_normal_initializer(stddev=0.01), #fine-tuning needed
                                             kernel_regularizer= regularizer)
    
    # print (last_layer)
    return (last_layer)
    # added logic ends - 2
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function  #komer 3
    # added logic -3 begins 
    # reshape the 4D tensors to 2D tensors
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    label = tf.reshape(correct_label, (-1,num_classes))

    # define cross_entropy_loss

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels=label))
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_constant = 0.01  # needed finetuning
    cross_entropy_loss = cross_entropy_loss + reg_constant * sum(reg_losses)
    
    # instantiate training operation with Adam Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_operation = optimizer.minimize(cross_entropy_loss)
    return (logits, training_operation, cross_entropy_loss)
    # added logic 3 - ends
    
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function #komer 4
    # added logic 4 - begins
    sess.run(tf.global_variables_initializer())
    print("Training in progress ...")
    print()
    all_losses = []
    for an_epoch in range(epochs):
        print("EPOCH {} ..".format(an_epoch))

        avg_epoch_loss = 0.
        num_epoch_runs = 0
        for an_img, a_label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict={input_image: an_img, correct_label: a_label, 
                                keep_prob: 0.5, learning_rate: 0.001})
            # print("Loss: {:.4f}".format(loss))
            avg_epoch_loss += loss
            num_epoch_runs += 1
        avg_epoch_loss /= num_epoch_runs
        
        print("Average EPOCH Loss: {:.4f}".format(avg_epoch_loss))
        all_losses.append(avg_epoch_loss)
        print()
    
    #print(all_losses)
    # added logic 4 - ends
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = '/data'
    runs_dir = './runs'
    
    EPOCHS = 50
    BATCH_SIZE = 10 # reduced from 16.
    
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
       # TODO: Build NN using load_vgg, layers, and optimize function # komer 5
                
        # Tensor Flow Placeholder Variables
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        # run the pipe line functions created
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function # komer 6
        tf.set_random_seed(786)
        sess.run(tf.global_variables_initializer())
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, input_image,
                correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples # komer 7
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        
        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
