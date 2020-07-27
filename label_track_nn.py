import numpy as np
import tensorflow as tf
import os
from scipy import ndimage
import sys
import matplotlib
from utils import gaussian_kernel, save_data, pre_process_cutout


list_path = '/home/su/Desktop/lastro/'  # path where the results will be written to
ones_path = '/home/su/Desktop/lastro/cutouts/ones/'  # directory of cutouts with tracks
zeros_path = '/home/su/Desktop/lastro/cutouts/zeros/'  # directory of cutouts without tracks
test_iterator = 0
image_size = 128  # cutout size: (image_size, image_size)
bin_size = 4
im_size = int(image_size/bin_size)  # binned cutout size: (image_size/bin_size, image_size/bin_size)
matplotlib.use('TkAgg')
os.environ['TF_KERAS'] = '1'
batch_size = 120  # batch size
training_epochs = 30  # number of training epochs


def calculate_length(total_length):
    """
    Calculate train and test length
    :param total_length: total number of data
    :return: train_length, test_length: number of files that will be used for training, testing
    """
    train_length = int(total_length*0.95)
    test_length = total_length - train_length
    return train_length, test_length


def prepare_data(ones_length, zeros_length):
    """
    Prepare training or test data and labels, discard used files from one_files and zero_files.
    :param ones_length: ones_train_length or ones_test_length
    :param zeros_length: zeros_train_length or zeros_test_length
    :return: labels, filename list of data
    """
    global one_files, zero_files
    labels = np.zeros(ones_length + zeros_length)
    labels[0:ones_length] = 1
    np.random.seed(np.random.randint(100))
    np.random.shuffle(labels)
    file_list = []
    on, zr = 0, 0
    # fill file_list based on labels array
    for k in range(0, len(labels)):
        if labels[k] == 0:
            file_list.append(zero_files[zr])
            zr += 1
        elif labels[k] == 1:
            file_list.append(one_files[on])
            on += 1
        else:
            print('There is a problem')

    # remove used files from one_files and zero_files
    zero_files = zero_files[zeros_length:]
    one_files = one_files[ones_length:]
    return labels, file_list


def load_train_data():
    """
    Fills train_images array with cutouts.
    """
    global train_images

    for k in range(0, train_length):
        if labels[k] == 1:
            train_im = np.load(ones_path+file_list1[k])
        elif labels[k] == 0:
            train_im = np.load(zeros_path+file_list1[k])
        else:
            print('There is something wrong')
            train_im = 0

        # filter and bin the cutout, scale it to [0, 1]
        train_im = ndimage.convolve(train_im, gaussian_kernel(5, 5))
        train_images[k] = pre_process_cutout(train_im, im_size, bin_size)


def load_test_data(ind):
    """
    Fill test_image with the cutout at the given index of file_list2
    :param ind: given index
    :return: file name of the image
    """
    global test_image
    global test_iterator

    test_filename = file_list2[test_iterator]
    if labels[ind] == 1:
        test_im = np.load(ones_path + test_filename)
    elif labels[ind] == 0:
        test_im = np.load(zeros_path + test_filename)
    else:
        print('There is something wrong')
        test_im = 0

    test_iterator += 1
    test_image = pre_process_cutout(test_im, im_size, bin_size)  # filter and bin the cutout, scale it to [0, 1]

    return test_filename


def add_block(dim, first_layer):
    """
    Add a block of convolutional layers to the model.
    :param dim: number of filters
    :param first_layer: True if it's the first layer of the network, false otherwise
    """
    if first_layer:
        model.add(tf.keras.layers.ZeroPadding2D((1, 1), input_shape=(im_size, im_size, 1)))
    else:
        model.add(tf.keras.layers.ZeroPadding2D((1, 1)))

    model.add(tf.keras.layers.Convolution2D(dim, (3, 3), activation='relu'))
    model.add(tf.keras.layers.ZeroPadding2D((1, 1)))
    model.add(tf.keras.layers.Convolution2D(dim, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))


def get_model():
    """
    Load model
    :return: model
    """
    global model
    model = tf.keras.Sequential()
    add_block(64, True)
    add_block(128, False)
    add_block(256, False)

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.1)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid',
                                    bias_initializer=tf.keras.initializers.Constant(np.log(ones_train_length /
                                                                                           zeros_train_length))))
    return model


# get files corresponding to cutouts
one_files = [f for f in os.listdir(ones_path) if os.path.isfile(os.path.join(ones_path, f))]
zero_files = [f for f in os.listdir(zeros_path) if os.path.isfile(os.path.join(zeros_path, f))]

# calculate train & test lengths
ones_train_length, ones_test_length = calculate_length(len(one_files))
zeros_train_length, zeros_test_length = calculate_length(len(zero_files))

np.random.seed(np.random.randint(100))
np.random.shuffle(one_files)
np.random.seed(np.random.randint(100))
np.random.shuffle(zero_files)

# prepare training data
(labels, file_list1) = prepare_data(ones_train_length, zeros_train_length)

train_length, test_length = (ones_train_length + zeros_train_length), (ones_test_length + zeros_test_length)
one_percentage = ones_train_length/train_length*100
zero_percentage = zeros_train_length/train_length*100

print('train_length: '+str(train_length), ' test_length: '+str(test_length))
print('1 train_length: '+str(ones_train_length), ' 1 test_length: '+str(ones_test_length))
print('0 train_length: '+str(zeros_train_length), ' 0 test_length: '+str(zeros_test_length))
print('1 percentage: ' + str(one_percentage), '%\n0 percentage: ' + str(zero_percentage) + ' %')

# calculate class weights
weight_for_0 = (1 / zeros_train_length)*train_length/2.0
weight_for_1 = (1 / ones_train_length)*train_length/2.0

class_weights = {0: weight_for_0, 1: weight_for_1}
print('class weights:', class_weights)

train_images = np.zeros((train_length, im_size, im_size))
test_image = np.zeros((1, im_size, im_size))

# get the model
model = get_model()
# initialize bias based on the ratio of ones to zeros, for a faster convergence
print('Initial bias: '+str(np.log(ones_train_length/zeros_train_length)))
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy', tf.keras.metrics.FalsePositives(name='fp'), tf.keras.metrics.FalseNegatives(name='fn')])

training_acc = []
# fill train_images with respect to the labels
load_train_data()

print("LOADING DATA FINISHED")
history = model.fit(np.expand_dims(train_images, axis=-1), labels, batch_size=batch_size,
                    epochs=training_epochs, class_weight=class_weights)

print("TESTING STARTS")
test_loss, test_acc, test_fp, test_fn = np.zeros(test_length), np.zeros(test_length), np.zeros(test_length), \
                                        np.zeros(test_length)

# prepare test data
(labels, file_list2) = prepare_data(ones_test_length, zeros_test_length)

wrong_predictions = {}
correct_satellite_predictions = []
predictions = {}

# test the network one cutout at a time
for k in range(0, test_length):
    fn = load_test_data(k)
    test_loss[k], test_acc[k], test_fp[k], test_fn[k] = \
        model.evaluate(np.expand_dims(np.expand_dims(test_image, axis=-1), axis=0), np.expand_dims(labels[k], axis=0),
                       verbose=0, batch_size=1)

    if int(test_acc[k]) == 0:
        pred = model.predict(np.expand_dims(np.expand_dims(test_image, axis=-1), axis=0), batch_size=1)
        wrong_predictions.update({fn: str(labels[k]) + ' ' + str(pred[0][0])})
    else:
        if labels[k] == 1:
            correct_satellite_predictions.append(fn)

print('Accuracy = ' + str(np.mean(test_acc)))
print('# of false positives: ' + str(np.sum(test_fp)))
print('# of false negatives: ' + str(np.sum(test_fn)))

# save wrong predictions, correct predictions of cutouts with tracks and the model itself
wr_pr_f = open(list_path+'dense_32/'+str(np.mean(test_acc))+'_wrong_predicted_list.txt', 'w+')
save_data(wrong_predictions, wr_pr_f)

c_s_p = open(list_path + 'correct_satellite_predictions.txt', 'w+')
c_s_p.write('\n'.join(correct_satellite_predictions))

wr_pr_f.close()
c_s_p.close()

model.save(list_path+'dense_32/model_'+str(np.mean(test_acc))+'.h5')
