import tensorflow as tf
import numpy as np

import os
import logging

from ae import AE
from vae import VAE

from vanilla_gan import VANILLA_GAN
from info_gan import INFO_GAN
from fm_gan import FM_GAN

from vae_vanilla_gan import VAE_VANILLA_GAN
from vae_vanilla_gan_improved import VAE_VANILLA_GAN_IMPROVED

def main(_):
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.log_device_placement = True

    #merged = tf.summary.merge_all()
    
    with tf.device('cpu:0'):
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.max_to_keep)

    ########## TRAIN ##########
    if FLAGS.is_train == True:
        with tf.Session(config=config) as sess:
            sess.run(init)

            #writer = tf.train.SummaryWriter(tensorboard_dir, sess.graph)
            train_batch_total = int(data.train.num_examples / FLAGS.batch_size)
            valid_batch_total = int(data.validation.num_examples/ FLAGS.batch_size)

            best_valid_total_cost = float('inf')
            best_model_idx = 0
            valid_non_improve_count = 0

            for epoch_idx in range(FLAGS.epoch):
                train_total_cost = 0
                valid_total_cost = 0
                
                ##### TRAIN #####
                for batch_idx in range(train_batch_total):
                    if ((batch_idx+1) % FLAGS.batch_logging_step == 0):
                        log_flag = True
                    else:
                        log_flag = False

                    if mnist_flag == True:
                        batch_xs, batch_ys = data.train.next_batch(FLAGS.batch_size)
                    elif mnist_flag == False:
                        batch_xs = data.train.next_batch(FLAGS.batch_size)
                    
                    if FLAGS.model == 'INFO_GAN':
                        cost_val = model.train_using_info(logger, sess, batch_xs, batch_ys, epoch_idx, batch_idx, train_batch_total, log_flag, FLAGS.keep_prob)
                    else:
                        cost_val = model.train(logger, sess, batch_xs, epoch_idx, batch_idx, train_batch_total, log_flag, FLAGS.keep_prob)

                    #_, cost_val, summary = sess.run([solver, cost, merged], feed_dict={X: batch_xs})
                    #writer.add_summary(summary, global_step=epoch_idx*tranin_total_batch+batch_idx)
                    #_, cost_val = sess.run([solver, cost], feed_dict={X: batch_xs})
                    train_total_cost += cost_val

                logger.debug('Epoch %.3i, Train loss: %.4E' % (epoch_idx, train_total_cost / train_batch_total))
                save_path = saver.save(sess, ckpt_path, global_step=epoch_idx) 


                ##### VALIDATION #####
                for batch_idx in range(valid_batch_total):
                    if ((batch_idx+1) % FLAGS.batch_logging_step == 0):
                        log_flag = True
                    else:
                        log_flag = False

                    if mnist_flag == True:
                        batch_xs, batch_ys = data.validation.next_batch(FLAGS.batch_size)
                    elif mnist_flag == False:
                        batch_xs = data.validation.next_batch(FLAGS.batch_size)

                    if FLAGS.model == 'INFO_GAN':
                        cost_val = model.inference_using_info(logger, sess, batch_xs, batch_ys, epoch_idx, batch_idx, valid_batch_total, log_flag, 1.0)
                    else:
                        cost_val = model.inference(logger, sess, batch_xs, epoch_idx, batch_idx, valid_batch_total, log_flag, 1.0)
                    valid_total_cost += cost_val

                logger.debug('Epoch %.3i, Valid loss: %.4E' % (epoch_idx, valid_total_cost / valid_batch_total))

                ### Update best_valid_total_cost ###
                if valid_total_cost < best_valid_total_cost:
                    best_valid_total_cost = valid_total_cost
                    valid_non_improve_count = 0
                    best_model_idx = epoch_idx 
                    logger.info("Best model idx : " + str(best_model_idx))

                    f = open(ckpt_dir+'/best_model_idx.txt','w')
                    f.write(str(best_model_idx))
                    f.close()
                else:
                    valid_non_improve_count += 1
                    logger.info("Valid cost has not been improved for %d epochs" % valid_non_improve_count)
                    if valid_non_improve_count == FLAGS.max_to_keep-1 and FLAGS.early_stop == True:
                        break

                ### Draw images ### 
                if mnist_flag == True:
                    sample_size = 16
                    if FLAGS.model == 'INFO_GAN':
                        _, samples = model.inference_with_output_using_info(logger, sess, data.test.images[:sample_size], data.test.labels[:sample_size], 0, 0, 1, False, 1.0)
                    else:
                        _, samples = model.inference_with_output(logger, sess, data.test.images[:sample_size], 0, 0, 1, False, 1.0)
                    fig = drawer.plot(samples)
                    plt.savefig(image_dir + '/{}.png'.format(str(epoch_idx).zfill(3)), bbox_inches='tight')


    ########## TEST ##########
    if FLAGS.dataset == "Music":
        with tf.Session(config=config) as sess:
            test_total_cost = 0
            top_k_accuracy = 0

            if FLAGS.is_train == False:
                if tf.train.get_checkpoint_state(ckpt_dir):
                    f = open(ckpt_dir+'/best_model_idx.txt','r')
                    best_model_idx = eval(f.readline().strip())

            saver.restore(sess, ckpt_path+"-"+str(best_model_idx))

            test_batch_total = int(data.test.num_examples / FLAGS.batch_size)
            test_logging_step = int(test_batch_total/10);

            for batch_idx in range(test_batch_total):
                batch_xs, batch_idxs = data.test.next_batch_with_idx(FLAGS.batch_size)

                if ((batch_idx+1) % FLAGS.batch_logging_step == 0):
                    log_flag = True
                else:
                    log_flag = False

                cost_val, top_k = model.inference_with_top_k(logger, sess, batch_xs, best_model_idx, batch_idx, test_batch_total, log_flag, 1.0, FLAGS.k)
                test_total_cost += cost_val

                values, indices = top_k 
                in_top_k = np.any(np.isclose(batch_idx, np.transpose(indices)), axis=0)
                top_k_accuracy += np.sum(in_top_k)

            logger.debug('Best model %.3i, Test loss: %.4E' % (best_model_idx, test_total_cost / test_batch_total))
            logger.info("Top %d accuracy : %.4E" % (FLAGS.k, top_k_accuracy/(FLAGS.batch_size*test_batch_total)))

if __name__ == '__main__':

    ########## SET ARGUMENTS ##########
    flags = tf.app.flags

    flags.DEFINE_string("model", "AE", "[RBM, AE, VAE, VANILLA_GAN, INFO_GAN, VAE_VANILLA_GAN, VAE_EB_GAN]")
    flags.DEFINE_string("dataset", "MNIST", "Dataset (MNIST,Music) for experiment [MNIST]")

    flags.DEFINE_boolean("is_train", False, "True for training, False for testing [Fasle]")
    flags.DEFINE_boolean("continue_train", None, "True to continue training from saved checkpoint. False for restarting. None for automatic [None]")

    flags.DEFINE_integer("gpu_id", 0, "GPU id [0]")

    flags.DEFINE_string("ae_h_dim_list", "[256]", "List of AE dimensions [256]")
    flags.DEFINE_integer("z_dim", 128, "Dimension of z [128]")
    flags.DEFINE_string("dis_h_dim_list", "[256]", "List of discriminator dimensions [256]")

    flags.DEFINE_integer("epoch", 300, "Epoch to train [300]")
    flags.DEFINE_float("learning_rate", 0.001, "Learning rate [0.001]")
    flags.DEFINE_float("keep_prob", 0.9, "Dropout keep probability [0.9]")
    flags.DEFINE_string("loss_type", "CE", "Loss type for reconsturction; CE for cross entropy, MSE for mean squared error [CE]")

    flags.DEFINE_integer("batch_size", 2048, "Batch size [2048]")
    flags.DEFINE_integer("batch_logging_step", 10, "Batch size [10]")
    flags.DEFINE_integer("k", 5, "k for top k measure [5]")

    flags.DEFINE_boolean("early_stop", True, "Early stop based on validation")
    flags.DEFINE_integer("max_to_keep", "11", "Maximum number of recent checkpoint files to keep; it can be used for early stop [11]")

    FLAGS = flags.FLAGS

    ########## DATA ##########
    ### todo : refactor mnist_flag to info_flag 
    if FLAGS.dataset == "MNIST":
        mnist_flag = True
        import matplotlib.pyplot as plt
        from utils import Drawer
        drawer = Drawer()

        from tensorflow.examples.tutorials.mnist import input_data
        data = input_data.read_data_sets("MNIST/data/", one_hot=True)
        input_dim = 28*28

    elif FLAGS.dataset == "Music":
        mnist_flag = False
        import data_loader
        data = data_loader.load_music_data()
        input_dim = data.train.dimension

    ########## BUILD MODEL ##########
    ### ckpt path should not contain '[' or ']'
    ae_h_dim_list_replaced = FLAGS.ae_h_dim_list.replace('[','').replace(']','').replace(',','-') 
    dis_h_dim_list_replaced = FLAGS.dis_h_dim_list.replace('[','').replace(']','').replace(',','-') 
    model_spec = 'm' + FLAGS.model + '_lr' + str(FLAGS.learning_rate) + '_e' + str(FLAGS.epoch) + '_keep' + str(FLAGS.keep_prob) + '_b' + str(FLAGS.batch_size) + '_ae' + ae_h_dim_list_replaced + '_z' + str(FLAGS.z_dim)

    ##### AE ##### 
    if FLAGS.model == 'AE': 
        model = AE(FLAGS.gpu_id, FLAGS.learning_rate, FLAGS.loss_type, input_dim, FLAGS.z_dim, eval(FLAGS.ae_h_dim_list))
    elif FLAGS.model == 'VAE':
        model = VAE(FLAGS.gpu_id, FLAGS.learning_rate, FLAGS.loss_type, input_dim, FLAGS.z_dim, eval(FLAGS.ae_h_dim_list))

    ##### GAN #####
    elif FLAGS.model == 'VANILLA_GAN':
        model = VANILLA_GAN(FLAGS.gpu_id, FLAGS.learning_rate, FLAGS.loss_type, input_dim, FLAGS.z_dim, eval(FLAGS.ae_h_dim_list), eval(FLAGS.dis_h_dim_list))
        model_spec += '_dis' + dis_h_dim_list_replaced
    elif FLAGS.model == 'INFO_GAN':
        model = INFO_GAN(FLAGS.gpu_id, FLAGS.learning_rate, FLAGS.loss_type, input_dim, FLAGS.z_dim, eval(FLAGS.ae_h_dim_list), eval(FLAGS.dis_h_dim_list))
        model_spec += '_dis' + dis_h_dim_list_replaced
    elif FLAGS.model == 'FM_GAN':
        model = FM_GAN(FLAGS.gpu_id, FLAGS.learning_rate, FLAGS.loss_type, input_dim, FLAGS.z_dim, eval(FLAGS.ae_h_dim_list), eval(FLAGS.dis_h_dim_list))
        model_spec += '_dis' + dis_h_dim_list_replaced

    ##### VAE_GAN #####
    elif FLAGS.model == 'VAE_VANILLA_GAN':
        model = VAE_VANILLA_GAN(FLAGS.gpu_id, FLAGS.learning_rate, FLAGS.loss_type, input_dim, FLAGS.z_dim, eval(FLAGS.ae_h_dim_list), eval(FLAGS.dis_h_dim_list))
        model_spec += '_dis' + dis_h_dim_list_replaced
    elif FLAGS.model == 'VAE_VANILLA_GAN_IMPROVED':
        model = VAE_VANILLA_GAN_IMPROVED(FLAGS.gpu_id, FLAGS.learning_rate, FLAGS.loss_type, input_dim, FLAGS.z_dim, eval(FLAGS.ae_h_dim_list), eval(FLAGS.dis_h_dim_list))
        model_spec += '_dis' + dis_h_dim_list_replaced

    ########## Make dir ##########
    ##### log #####
    log_dir = os.path.join(*[FLAGS.dataset, "log", model_spec])
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    fh = logging.FileHandler(os.path.join(log_dir, 'log'))

    fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    sh.setFormatter(fmt)
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)

    ##### ckpt #####
    ckpt_dir = os.path.join(*[FLAGS.dataset, "ckpt", model_spec])
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_path = os.path.join(ckpt_dir, "model_ckpt")

    ##### tensorboard ##### not used 
    tensorboard_dir = os.path.join(*[FLAGS.dataset, "tensorboard", model_spec])
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    
    ##### MNIST image #####
    if mnist_flag == True:
        image_dir = os.path.join("MNIST/images_visualized", model_spec)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

    tf.app.run()
