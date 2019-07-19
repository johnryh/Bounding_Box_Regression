import tensorflow as tf
from BB_Image_generator import *
from config import *

def parse_func(img, label):

    #cast to proper type
    img = tf.cast(img, dtype=tf.float32)
    label = tf.cast(label, tf.float32)
    label = label/img_h

    return img, label


def build_input_pipline(batch_size):
    ds_train = tf.data.Dataset.from_generator(generate_img, (tf.float32, tf.float32), (tf.TensorShape([100, 100]), tf.TensorShape([4])))
    ds_train = ds_train.map(parse_func, num_parallel_calls=12)

    ds_train = ds_train.repeat().batch(batch_size).prefetch(batch_size * 3) # add shuffling
    iterator_train = ds_train.make_one_shot_iterator()

    return iterator_train.get_next()


if __name__ == '__main__':
    img_batch, label_batch = build_input_pipline(batch_size)
    config = tf.ConfigProto(allow_soft_placement=True)
    plt.ion()

    with tf.Session(config=config) as sess:
        for _ in tqdm(range(10000)):
            img_l, label_l = sess.run([img_batch, label_batch])
            print(img_l.shape)
            print(label_l.shape)
            for sample in range(batch_size):
                img = img_l[sample,:,:]
                label = label_l[sample,:]
                plt.title('x:{}, y:{}, w:{}, h:{}'.format(label[0], label[1], label[2], label[3]))
                plt.imshow(img)
                plt.pause(1)