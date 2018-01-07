
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
import scipy.io
import pdb
import time
import os


# In[2]:

import scipy.misc, sys


# In[3]:

tf.reset_default_graph()


# In[4]:

STYLE_LAYERS = ('conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1')
CONTENT_LAYER = 'conv4_2'

MEAN_PIXEL = np.array([ 123.68 , 116.779, 103.939])

CONTENT_WEIGHT = 7.5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 2e2

LEARNING_RATE = 1e-3
WEIGHTS_INIT_STDEV = .1
BATCH_SIZE = 4
NUM_EPOCHS = 2


# In[5]:

def _weights(vgg_layers, layer, expected_layer_name):
    """ Return the weights and biases already trained by VGG
    """
    W = vgg_layers[0][layer][0][0][2][0][0]
    b = vgg_layers[0][layer][0][0][2][0][1]
    layer_name = vgg_layers[0][layer][0][0][0][0]
    assert layer_name == expected_layer_name
    return W, b.reshape(b.size)

def _conv2d_relu(vgg_layers, prev_layer, layer, layer_name):
    """ Return the Conv2D layer with RELU using the weights, biases from the VGG
    model at 'layer'.
    Inputs:
        vgg_layers: holding all the layers of VGGNet
        prev_layer: the output tensor from the previous layer
        layer: the index to current layer in vgg_layers
        layer_name: the string that is the name of the current layer.
                    It's used to specify variable_scope.
    Output:
        relu applied on the convolution.
    Note that you first need to obtain W and b from vgg-layers using the function
    _weights() defined above.
    W and b returned from _weights() are numpy arrays, so you have
    to convert them to TF tensors using tf.constant.
    Note that you'll have to do apply relu on the convolution.
    Hint for choosing strides size: 
        for small images, you probably don't want to skip any pixel
    """
    with tf.variable_scope(layer_name) as scope:
        W, b = _weights(vgg_layers, layer, layer_name)
        W = tf.constant(W, name='weights')
        b = tf.constant(b, name='bias')
        conv2d = tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv2d + b)

def _avgpool(prev_layer):
    """ Return the average pooling layer. The paper suggests that average pooling
    actually works better than max pooling.
    Input:
        prev_layer: the output tensor from the previous layer
    Output:
        the output of the tf.nn.avg_pool() function.
    Hint for choosing strides and kszie: choose what you feel appropriate
    """
    return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
                          padding='SAME', name='avg_pool_')

def load_vgg(path, input_image):
    """ Load VGG into a TensorFlow model.
    Use a dictionary to hold the model instead of using a Python class
    """
    vgg = scipy.io.loadmat(path)
    vgg_layers = vgg['layers']

    graph = {}
    graph['conv1_1']  = _conv2d_relu(vgg_layers, input_image, 0, 'conv1_1')
    graph['conv1_2']  = _conv2d_relu(vgg_layers, graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = _avgpool(graph['conv1_2'])
    graph['conv2_1']  = _conv2d_relu(vgg_layers, graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2']  = _conv2d_relu(vgg_layers, graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = _avgpool(graph['conv2_2'])
    graph['conv3_1']  = _conv2d_relu(vgg_layers, graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2']  = _conv2d_relu(vgg_layers, graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3']  = _conv2d_relu(vgg_layers, graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4']  = _conv2d_relu(vgg_layers, graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = _avgpool(graph['conv3_4'])
    graph['conv4_1']  = _conv2d_relu(vgg_layers, graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2']  = _conv2d_relu(vgg_layers, graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3']  = _conv2d_relu(vgg_layers, graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4']  = _conv2d_relu(vgg_layers, graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = _avgpool(graph['conv4_4'])
    graph['conv5_1']  = _conv2d_relu(vgg_layers, graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2']  = _conv2d_relu(vgg_layers, graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3']  = _conv2d_relu(vgg_layers, graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4']  = _conv2d_relu(vgg_layers, graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = _avgpool(graph['conv5_4'])
    
    return graph


# In[ ]:




# In[ ]:




# In[ ]:




# In[7]:

style_image = scipy.misc.imread('starry_night.jpg')
print(style_image.shape)
style_image = scipy.misc.imresize(style_image, (256,256,3))
print(style_image.shape)
style_image = style_image - MEAN_PIXEL
style_image = np.asarray(style_image, np.float32)
style_image = np.expand_dims(style_image, axis = 0)
print(style_image.shape)


# In[8]:

input_image = tf.placeholder(tf.float32, shape=(None,256,256,3), name='style_image')
vgg_net = load_vgg('imagenet-vgg-verydeep-19.mat', input_image)


# In[9]:

style_features = {}
with tf.Session() as sess :
    for layer in STYLE_LAYERS :
        tmp = sess.run(vgg_net[layer], feed_dict = {input_image : style_image})
        tmp = np.reshape(tmp, (-1,tmp.shape[3]))
        gram = np.matmul(tmp.T, tmp)/float(tmp.size)
        style_features[layer] = gram


# In[10]:

style_features['conv1_1'].shape


# In[11]:

def load_style_net(input_image):
    graph = {}
    graph['conv1'] = _conv_layer(input_image, 32, 9, 1)
    graph['conv2'] = _conv_layer(graph['conv1'], 64, 3, 2)
    graph['conv3'] = _conv_layer(graph['conv2'], 128, 3, 2)
    graph['resid1'] = _residual_block(graph['conv3'], 3)
    graph['resid2'] = _residual_block(graph['resid1'], 3)
    graph['resid3'] = _residual_block(graph['resid2'], 3)
    graph['resid4'] = _residual_block(graph['resid3'], 3)
    graph['resid5'] = _residual_block(graph['resid4'], 3)
    graph['conv_t1'] = _conv_tranpose_layer(graph['resid5'], 64, 3, 2)
    graph['conv_t2'] = _conv_tranpose_layer(graph['conv_t1'], 32, 3, 2)
    graph['conv_t3'] = _conv_layer(graph['conv_t2'], 3, 9, 1, relu=False)
    graph['preds'] = tf.nn.tanh(graph['conv_t3']) * 150.0 + 255./2
    return graph

def _conv_layer(net, num_filters, filter_size, strides, relu=True):
    weights_init = _conv_init_vars(net, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]
    net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')
    net = _instance_norm(net)
    if relu:
        net = tf.nn.relu(net)

    return net

def _conv_tranpose_layer(net, num_filters, filter_size, strides):
    weights_init = _conv_init_vars(net, num_filters, filter_size, transpose=True)

    batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
    new_rows, new_cols = int(rows * strides), int(cols * strides)
    # new_shape = #tf.pack([tf.shape(net)[0], new_rows, new_cols, num_filters])

    new_shape = [batch_size, new_rows, new_cols, num_filters]
    tf_shape = tf.stack(new_shape)
    strides_shape = [1,strides,strides,1]
    net = tf.nn.conv2d_transpose(net, weights_init, new_shape, strides_shape, padding='SAME')
    net = _instance_norm(net)
    return tf.nn.relu(net)

def _residual_block(net, filter_size=3):
    tmp = _conv_layer(net, 128, filter_size, 1)
    return net + _conv_layer(tmp, 128, filter_size, 1, relu=False)

def _instance_norm(net, train=True):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
    return scale * normalized + shift

def _conv_init_vars(net, out_channels, filter_size, transpose=False):
    _, rows, cols, in_channels = [i.value for i in net.get_shape()]
    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=WEIGHTS_INIT_STDEV, seed=1), dtype=tf.float32)
    return weights_init


# In[12]:

X_content = tf.placeholder(tf.float32, shape=(BATCH_SIZE,256,256,3), name='input_batch')
X_content = X_content - MEAN_PIXEL
content_features = load_vgg('imagenet-vgg-verydeep-19.mat', X_content)


# In[13]:

styled_image = load_style_net(X_content/255.)
X_preds = styled_image['preds'] - MEAN_PIXEL
net = load_vgg('imagenet-vgg-verydeep-19.mat', X_preds)


# In[14]:

tmp = net[CONTENT_LAYER].get_shape().as_list()
content_size = tmp[0]*tmp[1]*tmp[2]*tmp[3]
content_loss = CONTENT_WEIGHT * (2 * tf.nn.l2_loss(content_features[CONTENT_LAYER]-net[CONTENT_LAYER]))/content_size


# In[15]:

style_loss = 0.0
for style_layer in STYLE_LAYERS:
    layer = net[style_layer]
    bs, height, width, filters = layer.get_shape().as_list()
    size = height * width * filters
    feats = tf.reshape(layer, (bs, height * width, filters))
    feats_T = tf.transpose(feats, perm=[0,2,1])
    grams = tf.matmul(feats_T, feats) / size
    style_gram = style_features[style_layer]
    style_losses = tf.add(style_loss, (2 * tf.nn.l2_loss(grams - style_gram)/style_gram.size))
style_loss = STYLE_WEIGHT * style_loss/ BATCH_SIZE


# In[16]:

# total variation denoising
batch_shape = [BATCH_SIZE, 256,256,3]
tmp = X_preds[:,1:,:,:].get_shape().as_list()[1:]
tv_y_size = tmp[0]*tmp[1]*tmp[2]
tmp = X_preds[:,:,1:,:].get_shape().as_list()[1:]
tv_x_size = tmp[0]*tmp[1]*tmp[2]
y_tv = tf.nn.l2_loss(X_preds[:,1:,:,:] - X_preds[:,:batch_shape[1]-1,:,:])
x_tv = tf.nn.l2_loss(X_preds[:,:,1:,:] - X_preds[:,:,:batch_shape[2]-1,:])
tv_loss = TV_WEIGHT * 2 * (x_tv/tv_x_size + y_tv/tv_y_size)/BATCH_SIZE


# In[17]:

total_loss = content_loss + style_loss + tv_loss


# In[18]:

optimizer =  tf.train.AdamOptimizer(LEARNING_RATE).minimize(total_loss)


# In[19]:

###################   DATASET  #############################
def next_batch() :
    files = os.listdir('./train2014')
    #print('Total images : ', len(files))
    files_rem = len(files)%BATCH_SIZE
    files = files[:-files_rem]
    print('Total images : ', len(files))
    for i in xrange(0,len(files),BATCH_SIZE) :
        tmp = np.zeros((BATCH_SIZE, 256, 256, 3))
        for j in xrange(BATCH_SIZE) :
            image = files[i+j]
            tmp2 = scipy.misc.imread('./train2014/'+image)
            tmp2 = scipy.misc.imresize(tmp2,(256,256,3))
            tmp2 = np.asarray(tmp2,dtype = np.float32)
            if len(tmp2.shape) != 3 :
                tmp[j,:,:,0] = tmp2[:,:]
                tmp[j,:,:,1] = tmp2[:,:]
                tmp[j,:,:,2] = tmp2[:,:]
            else :
                tmp[j,:,:,:] = tmp2[:,:,:]
        
        yield np.asarray(tmp, dtype = np.float32)


# In[20]:

saver = tf.train.Saver()


# In[23]:

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    i = 0
    for epoch in xrange(NUM_EPOCHS) :
        print("EPOCH : ",epoch)
        tmp = 0
        for X_batch in next_batch() :
            _,l = sess.run([optimizer,total_loss], feed_dict = {'input_batch:0' : X_batch})
            if i%100 == 0:
                print('total_loss in batch No.',i,l/BATCH_SIZE)
            i += 1
            if i%5000 == 1 :
                tmp = sess.run(styled_image, feed_dict = {'input_batch:0' : X_batch})
                tmp = tmp['preds']
                scipy.misc.imsave('./samples/_'+str(epoch)+'_'+str(i)+'_1.png',tmp[0])
                scipy.misc.imsave('./samples/_'+str(epoch)+'_'+str(i)+'_2.png',tmp[1])
                scipy.misc.imsave('./samples/_'+str(epoch)+'_'+str(i)+'_3.png',tmp[2])
                scipy.misc.imsave('./samples/_'+str(epoch)+'_'+str(i)+'_4.png',tmp[3])
        saver.save(sess, './models/model',global_step=epoch)
                


# In[ ]:




# In[ ]:



