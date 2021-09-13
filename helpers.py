import tensorflow as tf
import numpy as np
import os

def byol_loss(p, z):
    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)
    sim = tf.reduce_sum(tf.multiply(p, z), axis=1)
    return 2 - 2 * tf.reduce_mean(sim)

class TDW_data:
    def __init__(self, path_training):
        images_1, images_2 = [], []
        #path = r"photoreal_80x80"
        images = os.listdir(path_training)
        for i in range(0,len(images),2):
            filepath1 = os.path.join(path_training,images[i])
            filepath2 = os.path.join(path_training,images[i+1])
            image1 = tf.io.read_file(filepath1)
            image1 = tf.image.decode_jpeg(image1)
            image2 = tf.io.read_file(filepath2)
            image2 = tf.image.decode_jpeg(image2)
            image1 = image1/255
            image2 = image2/255
            images_1.append(image1)
            images_2.append(image2)
        
        permutation = np.random.permutation(len(images_1))
        self.images_1 = [images_1[el] for el in permutation]
        self.images_2 = [images_2[el] for el in permutation]
        self.num_train_images, self.num_test_images = len(self.images_1), len(self.images_1)

    def get_batch_training(self, batch_id, batch_size):    
        x_batch_1 = tf.stack(self.images_1[batch_id*batch_size:(batch_id+1)*batch_size])
        x_batch_2 = tf.stack(self.images_2[batch_id*batch_size:(batch_id+1)*batch_size])
        return x_batch_1, x_batch_2
