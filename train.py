from build_input_pipeline import *
from network_utils import *
from tqdm import tqdm
import numpy as np
import os, time
import cv2

if __name__ == '__main__':

    tf.reset_default_graph()

    img_batch, label_batch = build_input_pipline(batch_size)


    #build model here
    img_batch = tf.reshape(img_batch, [batch_size, img_h, img_w, 1])
    label_batch = tf.reshape(label_batch, [batch_size, 4])

    print(img_batch, label_batch)
    model = BB_Regressor(img_batch, label_batch)

    merged = tf.summary.merge_all()


    run_metadata = tf.RunMetadata()
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        train_writer = tf.summary.FileWriter('runs/{}/logs/'.format(exp_name), sess.graph)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        print('Session Initiated')


        with tqdm(total=int(total_iter_num), unit='it') as pbar:
            train_start_time = time.time()

            plt.ion()
            for iter in range(total_iter_num):
                iter_start_time = time.time()

                _, img_l, labels_l, logits_l, loss, summary = sess.run([model.apply_grad, model.img, model.label, model.logits, model.loss, merged])

                #iter_per_sec = 1/(time.time() - iter_start_time)
                train_writer.add_summary(summary, iter)

                if iter % 10 == 0: pbar.set_postfix({'it_ins/s':'{:4.2f}, loss:{}'.format(1, loss)})
                pbar.update(1)

                if iter % 100 == 0:
                    curr_img = np.repeat(img_l[0, :,:, :], axis=2, repeats=3)
                    curr_label = labels_l[0,:]*img_h
                    curr_logit = logits_l[0,:]*img_h
                    print(curr_img.shape)
                    print(curr_label)
                    print(curr_logit)

                    # draw boxes
                    cv2.rectangle(curr_img, (curr_label[0], curr_label[1]), (curr_label[0]+curr_label[2], curr_label[1]+curr_label[3]), (0,255,0),1)
                    cv2.rectangle(curr_img, (curr_logit[0], curr_logit[1]), (curr_logit[0]+curr_logit[2], curr_logit[1]+curr_logit[3]), (255,0,0),1)

                    plt.imshow(curr_img)
                    plt.pause(0.1)