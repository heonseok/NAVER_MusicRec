import tensorflow as tf
import numpy as np
from base_model import BaseModel
from utils import get_random_normal 

class INFO_GAN(BaseModel):
    def __init__(self, gpu_id, learning_rate, loss_type, input_dim, z_dim, ae_h_dim_list, dis_h_dim_list):
        super(INFO_GAN, self).__init__(gpu_id, learning_rate, loss_type, input_dim, z_dim) 

        self.dec_h_dim_list = [*list(reversed(ae_h_dim_list))]
        self.dis_h_dim_list = dis_h_dim_list

        self.build_model()

    def build_model(self):
        with tf.device('/gpu:%d' % self.gpu_id):
            ### Placeholder ###
            self.X = tf.placeholder(tf.float32, [None, self.input_dim])
            self.y = tf.placeholder(tf.float32, [None, 10])
            self.k = tf.placeholder(tf.int32)
            self.z = tf.placeholder(tf.float32, [None, self.z_dim])
            self.keep_prob = tf.placeholder(tf.float32)
            
            ### Generating ###
            self.gen_X_logit = self.decoder(tf.concat([self.z,self.y], 1), self.dec_h_dim_list, self.input_dim, self.keep_prob, False)
            self.gen_X = tf.nn.sigmoid(self.gen_X_logit)
            self.output = tf.nn.sigmoid(self.gen_X_logit)

            ### Discriminating ###
            dis_logit_real = self.discriminator(tf.concat([self.X, self.y], 1), self.dis_h_dim_list, 1, self.keep_prob, False)
            dis_logit_fake = self.discriminator(tf.concat([self.gen_X, self.y], 1), self.dis_h_dim_list, 1, self.keep_prob, True)
            self.dec_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_logit_fake, labels=tf.ones_like(dis_logit_fake))) 

            self.dis_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_logit_real, labels=tf.ones_like(dis_logit_real))) 
            self.dis_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_logit_fake, labels=tf.zeros_like(dis_logit_fake))) 
            
            ### Loss ###
            self.dec_loss = self.dec_loss_fake
            self.dis_loss = self.dis_loss_real + self.dis_loss_fake

            ### Theta ###
            dec_theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dec')
            dis_theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dis')

            ### Solver ###
            self.dec_solver = tf.train.AdamOptimizer(self.learning_rate).minimize(self.dec_loss, var_list=dec_theta)
            self.dis_solver = tf.train.AdamOptimizer(self.learning_rate).minimize(self.dis_loss, var_list=dis_theta)


    def train_using_info(self, logger, sess, batch_xs, batch_ys, epoch_idx, batch_idx, batch_total, log_flag, keep_prob):
        random_z = get_random_normal(batch_xs.shape[0], self.z_dim)

        for i in range(5):
            _, dis_loss_val = sess.run([self.dis_solver, self.dis_loss], feed_dict={self.X: batch_xs, self.y: batch_ys, self.keep_prob: keep_prob, self.z: random_z})
        _, dec_loss_val = sess.run([self.dec_solver, self.dec_loss], feed_dict={self.y: batch_ys, self.keep_prob: keep_prob, self.z: random_z})

        total_loss_val = dis_loss_val + dec_loss_val
        if log_flag == True:
            logger.debug('Epoch %.3i, Batch[%.3i/%i], Dis loss : %.4E, Dec loss : %.4E, Train loss: %.4E' % (epoch_idx, batch_idx + 1, batch_total, dis_loss_val, dec_loss_val, total_loss_val))

        return total_loss_val

    def inference_using_info(self, logger, sess, batch_xs, batch_ys, epoch_idx, batch_idx, batch_total, log_flag, keep_prob):
        random_z = get_random_normal(batch_xs.shape[0], self.z_dim)

        dis_loss_val, dec_loss_val = sess.run([self.dis_loss, self.dec_loss], feed_dict={self.X: batch_xs, self.y: batch_ys, self.keep_prob: keep_prob, self.z: random_z})

        total_loss_val = dis_loss_val + dec_loss_val
        if log_flag == True:
            logger.debug('Epoch %.3i, Batch[%.3i/%i], Dis loss : %.4E, Dec loss : %.4E, Valid loss: %.4E' % (epoch_idx, batch_idx + 1, batch_total, dis_loss_val, dec_loss_val, total_loss_val))

        return total_loss_val

    def inference_with_output_using_info(self, logger, sess, batch_xs, batch_ys, epoch_idx, batch_idx, batch_total, log_flag, keep_prob):
        random_z = get_random_normal(batch_xs.shape[0], self.z_dim)

        dis_loss_val, dec_loss_val, output_val = sess.run([self.dis_loss, self.dec_loss, self.output], feed_dict={self.X: batch_xs, self.y: batch_ys, self.keep_prob: keep_prob, self.z: random_z})

        total_loss_val = dis_loss_val + dec_loss_val
        if log_flag == True:
            logger.debug('Epoch %.3i, Batch[%.3i/%i], Dis loss : %.4E, Dec loss : %.4E, Test loss: %.4E' % (epoch_idx, batch_idx + 1, batch_total, dis_loss_val, dec_loss_val, total_loss_val))

        return total_loss_val, output_val
