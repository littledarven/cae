kernel_sizeimport tensorflow as tf
import numpy as np


sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

f = h5py.File('train.h5')
train = np.array(f.get(f.keys()[0]))

learning_rate = 0.0005
epochs = 500
batch_size = 512

x = tf.placeholder(tf.float32, [None, 36425], name='InputData')
input_layer = tf.reshape(x, shape=[-1, 235, 155, 1])

def upsample(input, name, factor=[2,2]):
    size = [int(input.shape[1]*factor[0]), int(input.shape * factor[1])]
    with tf.name_scope(name):
        out = tf.image.resize_bilinear(input, size=size, align_corners=None, name=None)
        return out

conv1 = tf.layers.conv2d(inputs = input_layer, num_filters = 256, kernel_size = [11,11], use_bias=True, padding = "same", activation = tf.nn.relu)
pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [3,3], stride = 2)

conv2 = tf.layers.conv2d(inputs = relu1, num_filters = 128, kernel_size = [5,5], use_bias=True, padding = 2, activation = tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [3,3], stride = 2)

conv3 = tf.layers.conv2d(inputs = input_layer, num_filters = 64, kernel_size = [3,3], use_bias=True ,padding = 1, activation = tf.nn.relu)
pool3 = tf.layers.max_pooling2d(inputs = conv3, pool_size = [3,3], stride = 2)

deconv1 = tf.layers.conv2d_transpose(inputs = output, num_filters = 64, kernel_size[3,3], padding = 1, activation = tf.nn.relu)
unpool1 = upsample(deconv1, name='unpool1', factor[2,2])

deconv2 = tf.layers.conv2d_transpose(inputs = deconv1, num_filters=128, kernel_size[5,5], padding=2, strides=[2,2], activation = tf.nn.relu)
unpool2 = upsample(deconv2, name='unpool2', factor[2,2])

deconv3 = tf.layers.conv2d_transpose(inputs = deconv1, num_filters=256, kernel_size[5,5], padding="same", strides=[2,2], activation = tf.nn.relu)
unpool3 = upsample(deconv1, name='unpool3', factor[2,2])

def network_train(x):
    prediction, cost = ConvAutoEncoder(x, 'ConvAutoEnc')
    with tf.name_scope('opt'):
        optimizer = tf.train.AdamOptimizer().minimize(cost)

    tf.summary.scalar("cost", cost)

    merged_summary_op = tf.summary.merge_all()

    n_epochs = 500
    with tf.device("/gpu:0"):
        init_op = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init_op)

        //writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        for epoch in range(n_epochs):
            avg_cost = 0
            //n_batches = int(mnist.train.num_examples / batch_size)
            for i in range(n_batches):
                //batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                //_, c, summary = sess.run([optimizer, cost, merged_summary_op], feed_dict={x: batch_x, y: batch_y})
                # Compute average loss
                //avg_cost += c / n_batches
                # write log
                writer.add_summary(summary, epoch * n_batches + i)

            # Display logs per epoch step
            print('Epoch', epoch+1, ' / ', n_epochs, 'cost:', avg_cost)
        with sess.as_default() as sess:
            print('Optimization Finished')
            print('Cost:', cost.eval({x: mnist.test.images}))
