import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from moviepy.editor import VideoFileClip
import numpy as np
import scipy.misc
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
    tf.saved_model.loader.load(sess=sess,tags=[vgg_tag], export_dir=vgg_path)
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    return (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
#tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    conv_1x1 = tf.layers.conv2d(vgg_layer7_out,num_classes,1,padding='same',
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    output = tf.layers.conv2d_transpose(conv_1x1,vgg_layer4_out.shape[-1],4,2,padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    output = tf.add(output,vgg_layer4_out)
    output = tf.layers.conv2d_transpose(output, vgg_layer3_out.shape[-1], 4, 2, padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    output = tf.add(output,vgg_layer3_out)
    output = tf.layers.conv2d_transpose(output, num_classes, 16, 8, padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    return output
#tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=correct_label,logits=logits))
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss
#tests.test_optimize(optimize)


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
    # TODO: Implement function
    sess.run(tf.global_variables_initializer())
    for i in range(epochs):
        for batch in get_batches_fn(batch_size):
            _,loss = sess.run([train_op,cross_entropy_loss],feed_dict={input_image: batch[0], correct_label: batch[1], keep_prob: 0.5, learning_rate:0.001})
            print('epoch:',i+1,'loss=',loss)
    pass
#tests.test_train_nn(train_nn)

def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    export_dir = './inference'
    has_pretrained_graph = False
    if os.path.exists(export_dir):
        has_pretrained_graph = True
    else:
        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    clip = VideoFileClip('driving.mp4')

    def pipeline(img):
        img=scipy.misc.imresize(img, image_shape)
        im_softmax = sess.run([output_mask],
                                          feed_dict={image_input: [img],keep_prob:1.0})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(img)
        street_im.paste(mask, box=None, mask=mask)

        return np.array(street_im)

    with tf.Session() as sess:
        if has_pretrained_graph:
            tf.saved_model.loader.load(sess, ['inference_fcn'], export_dir)
            vgg_input_tensor_name = 'image_input:0'
            vgg_keep_prob_tensor_name = 'keep_prob:0'
            output_mask_tensor_name = 'output_mask:0'
            graph = tf.get_default_graph()
            image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
            keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
            output_mask = graph.get_tensor_by_name(output_mask_tensor_name)
            new_clip = clip.fl_image(pipeline)
            # write to file
            new_clip.write_videofile('result.mp4')

        else:
            # Path to vgg model
            vgg_path = os.path.join(data_dir, 'vgg')
            # Create function to get batches
            get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

            # OPTIONAL: Augment Images for better results
            #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

            # TODO: Build NN using load_vgg, layers, and optimize function
            image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess,vgg_path)
            output = layers(vgg_layer3_out=layer3_out,vgg_layer4_out=layer4_out,vgg_layer7_out=layer7_out,num_classes=num_classes)
            correct_labels = tf.placeholder(tf.float32, shape=[None,None,None, num_classes])
            learning_rate = tf.placeholder(tf.float32)
            logits, train_op, cross_entropy_loss = optimize(output,correct_labels,learning_rate,num_classes)
            train_nn(sess,8,12,get_batches_fn,train_op,cross_entropy_loss,image_input,correct_labels,keep_prob,learning_rate)
            output_mask = tf.nn.softmax(logits,name='output_mask')
            builder.add_meta_graph_and_variables(sess,['inference_fcn'])
            # TODO: Train NN using the train_nn function

            # TODO: Save inference data using helper.save_inference_samples
            input_image = image_input
            helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
            #print trained image IoU
            helper.iou_output(sess,logits,keep_prob,input_image,data_dir,image_shape,num_classes)
        # OPTIONAL: Apply the trained model to a video
    if not has_pretrained_graph:
        builder.save()

if __name__ == '__main__':
    run()
