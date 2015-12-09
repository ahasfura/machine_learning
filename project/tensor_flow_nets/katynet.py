'''
the katynet
out = fc(pool(conv(in)))
'''

# Import MINST data
import datetime
import input_data
import tensorflow as tf
import argparse
import time

# Create model
def conv2d(img, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME'),b))

def max_pool(img, k):
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def mean_pool(img, k):
    return tf.nn.avg_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def build_net(_X, _weights, _biases, _dropout, _pool_dim, _mean=True, _is_dropout=True, _is_relu=True):
    # Reshape input picture
    _X = tf.reshape(_X, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'])
    # Max Pooling (down-sampling)
    if _mean:
        conv1 = mean_pool(conv1, k=_pool_dim)
    else:
        conv1 = max_pool(conv1, k=_pool_dim)
    # Apply Dropout
    if _is_dropout:
        conv1 = tf.nn.dropout(conv1, _dropout)

    # Fully connected layer
    #dense1 = tf.reshape(conv1, [-1, _weights['wd1'].get_shape().as_list()[0]]) # Reshape conv2 output to fit dense layer input
    dense1 = tf.reshape(conv1, [-1, _weights['out'].get_shape().as_list()[0]]) # Reshape conv2 output to fit dense layer input
    '''
    if _is_relu:
        dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1'])) # Relu activation
    else:
        dense1 = tf.nn.tanh(tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1'])) # tanh activation
    if _is_dropout:
        dense1 = tf.nn.dropout(dense1, _dropout) # Apply Dropout
    '''

    # Output, class prediction
    out = tf.add(tf.matmul(dense1, _weights['out']), _biases['out'])
    return out

def write_out_data(filename, params, final_acc, time_elapsed):
    with open('../run_outputs/{}.txt'.format(filename), 'a') as f:
        f.write('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
        f.write(str(datetime.datetime.now())+'\n')
        f.write(str(params.keys())+'\n')
        f.write(str(params.values())+'\n')
        f.write("final accuracy: {}, time taken: {}\n".format(final_acc, time_elapsed))
        f.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Setup deep net')


    parser.add_argument("--run_name", required=True, type=str, help="name of run")
    parser.add_argument("--filter_size", type=int, default=9, help="size of convolve filter")
    parser.add_argument("--num_filters", type=int, default=20, help="number of filters")
    parser.add_argument("--learning_rate", type=float, default=.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--pool_dim", choices=[2, 4, 6, 8, 10], type=int, default=2, help="how big are the pooling layer dimensions")
    parser.add_argument("--dropout_rate", type=float, default=.75, help="set dropout rate")
    parser.add_argument("--training_iters", type=int, default=100000, help="how many iterations for the training phase?")
    #parser.add_argument("--fc_size", type=int, default=1024, help="how many hidden nodes for fully connected layer?")

    parser.add_argument("--relu", dest="relu", action="store_true", help="use relu")
    parser.add_argument("--tanh", dest="relu", action="store_false", help="use tanh")
    parser.set_defaults(relu=True)

    parser.add_argument("--mean", dest="mean", action="store_true", help="use mean pooling")
    parser.add_argument("--max", dest="mean", action="store_false", help="use max pooling")
    parser.set_defaults(mean=False)

    parser.add_argument("--dropout", dest="use_dropout", action="store_true", help="use dropout?")
    parser.add_argument("--no_dropout", dest="use_dropout", action="store_false", help="use dropout?")
    parser.set_defaults(use_dropout=True)

    parser.add_argument("--save_run", dest="save_run", action="store_true", help="save run")
    parser.add_argument("--dont_save_run", dest="save_run", action="store_false", help="don't save run")
    parser.set_defaults(save_run=True)

    args = parser.parse_args()

    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # Parameters
    learning_rate = args.learning_rate
    training_iters = args.training_iters
    batch_size = args.batch_size
    display_step = 10
    is_relu = args.relu
    is_dropout = args.use_dropout
    is_mean = args.mean
    filter_size = args.filter_size
    num_filters = args.num_filters
    pool_dim = args.pool_dim
    image_size = 28
    #fc_size = args.fc_size

    # Network Parameters
    n_input = 784 # MNIST data input (img shape: 28*28)
    n_classes = 10 # MNIST total classes (0-9 digits)
    dropout = args.dropout_rate # Dropout, probability to keep units

    # tf Graph input
    x = tf.placeholder(tf.types.float32, [None, n_input])
    y = tf.placeholder(tf.types.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.types.float32) #dropout (keep probability)
    # Store layers weight & bias
    weights = {
    'wc1': tf.Variable(tf.random_normal([filter_size, filter_size, 1, num_filters])), # 5x5 conv, 1 input, 32 outputs
    #'wd1': tf.Variable(tf.random_normal([(image_size/pool_dim)*(image_size/pool_dim)*num_filters, fc_size])), # fully connected, 14*14*32 inputs, 1024 outputs
    #'out': tf.Variable(tf.random_normal([fc_size, n_classes])) # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([(image_size/pool_dim)*(image_size/pool_dim)*num_filters, n_classes])) # 1024 inputs, 10 outputs (class prediction)
    }

    biases = {
    'bc1': tf.Variable(tf.random_normal([num_filters])),
    #'bd1': tf.Variable(tf.random_normal([fc_size])),
    #'out': tf.Variable(tf.random_normal([n_classes]))
    'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Construct model
    pred = build_net(x, weights, biases, keep_prob, pool_dim, _mean=is_mean, _is_dropout=is_dropout, _is_relu=is_relu)


    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.types.float32))

    # Initializing the variables
    init = tf.initialize_all_variables()

    # start timer
    start = time.time()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
            if step % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)
            step += 1

        print "Optimization Finished!"
        # Calculate accuracy for 256 mnist test images
        final_acc = sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.})
        print "Testing Accuracy:", str(final_acc)
    
    end = time.time()

    if args.save_run:
        write_out_data(args.run_name[:-3], {'number of layers':len(weights), 'learning_rate':learning_rate, 'training_iters':training_iters, 'batch_size':batch_size, 'filter_size':filter_size, 'num_filters':num_filters, 'pool_dim':pool_dim, 'dropout':dropout, 'mean pool?':is_mean, 'dropout?':is_dropout, 'relu?': is_relu}, final_acc, end - start)
