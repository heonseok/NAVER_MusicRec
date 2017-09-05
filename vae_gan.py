import tensorflow as tf
import numpy as np
import os
import logging
from utils import sample_z
from utils import kl_divergence_normal_distribution
from base_model import BaseModel

class VAE_GAN(BaseModel):
    def __init__(self, logger, gpu_id, learning_rate, loss_type, input_dim, z_dim, ae_h_dim_list, dis_h_dim_list):
        super(VAE_GAN, self).__init__(logger, gpu_id, learning_rate, loss_type, input_dim, z_dim)

        self.enc_h_dim_list = ae_h_dim_list
        self.dec_h_dim_list = [*list(reversed(ae_h_dim_list))]
        self.dis_h_dim_list = dis_h_dim_list

        self.build_model()

    def build_model(self):
        with tf.device('/gpu:%d' % self.gpu_id):
            self.X = tf.placeholder(tf.float32, [None, self.input_dim])
            self.k = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32)

            ### Encoding ### 
            self.z_mu, self.z_logvar = self.encoder(self.X, self.enc_h_dim_list, self.z_dim, self.keep_prob)
            self.z = sample_z(self.z_mu, self.z_logvar)

            ### Decoding/Generating ###
            self.recon_X_logit = self.decoder(self.z, self.dec_h_dim_list, self.input_dim, self.keep_prob, False)
            gen_X_logit = self.decoder(tf.random_normal(tf.shape(self.z)), self.dec_h_dim_list, self.input_dim, self.keep_prob, True)
            self.recon_X = tf.nn.sigmoid(self.recon_X_logit)
            gen_X = tf.nn.sigmoid(gen_X_logit)

            ### Discriminating ###
            dis_logit_real, dis_prob_real = self.discriminator(self.X, self.dis_h_dim_list, 1, self.keep_prob, False)
            #dis_logit_real, dis_prob_real = self.discriminator(self.recon_X, self.dis_h_dim_list, 1, self.keep_prob, False)
            dis_logit_fake, dis_prob_fake = self.discriminator(gen_X, self.dis_h_dim_list, 1, self.keep_prob, True)


            self.logger.info([x.name for x in tf.global_variables()])
            print([x.name for x in tf.global_variables() if 'enc' in x.name])
            print([x.name for x in tf.global_variables() if 'dec' in x.name])
            print([x.name for x in tf.global_variables() if 'dis' in x.name])

            enc_theta = ([x for x in tf.global_variables() if 'enc' in x.name])
            dec_theta = ([x for x in tf.global_variables() if 'dec' in x.name])
            dis_theta = ([x for x in tf.global_variables() if 'dis' in x.name])

            #cost = tf.reduce_mean(tf.square(X-output))
            ### Loss ###
            #self.recon_loss = tf.losses.mean_squared_error(self.X, self.recon_X)
            self.recon_loss = self.recon_loss() 
            self.kl_loss = kl_divergence_normal_distribution(self.z_mu, self.z_logvar)
            
            self.dec_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_logit_fake, labels=tf.ones_like(dis_logit_fake)))

            self.dis_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_logit_real, labels=tf.ones_like(dis_logit_real)))
            self.dis_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_logit_fake, labels=tf.zeros_like(dis_logit_fake)))

            self.enc_loss = self.recon_loss + self.kl_loss
            self.dec_loss = self.dec_loss_fake + self.recon_loss 
            self.dis_loss = self.dis_loss_real + self.dis_loss_fake
            #cost_summary = tf.summary.scalar('cost', cost)

            #self.total_loss = self.enc_loss + self.dec_loss + self.dis_loss 
            self.enc_solver = tf.train.AdamOptimizer(self.learning_rate).minimize(self.enc_loss, var_list=enc_theta)
            self.dec_solver = tf.train.AdamOptimizer(self.learning_rate).minimize(self.dec_loss, var_list=dec_theta)
            self.dis_solver = tf.train.AdamOptimizer(self.learning_rate).minimize(self.dis_loss, var_list=dis_theta)

            """
            self.enc_solver = tf.train.AdamOptimizer(self.learning_rate).minimize(self.enc_loss, var_list=enc_theta)
            self.dec_solver = tf.train.AdamOptimizer(self.learning_rate).minimize(self.dec_loss, var_list=dec_theta)
            self.dis_solver = tf.train.AdamOptimizer(self.learning_rate).minimize(self.dis_loss, var_list=dis_theta)
            """
            """
            self.enc_solver = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.enc_loss, var_list=enc_theta)
            self.dec_solver = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.dec_loss, var_list=dec_theta)
            self.dis_solver = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.dis_loss, var_list=dis_theta)
            """
            ### Recommendaiton metric ###
        with tf.device('/cpu:0'):
            self.top_k_op = tf.nn.top_k(self.recon_X, self.k)
        

    def train(self, sess, batch_xs, epoch_idx, batch_idx, batch_total, log_flag, keep_prob):
        for i in range(5):
            _, dis_loss_val = sess.run([self.dis_solver, self.dis_loss], feed_dict={self.X: batch_xs, self.keep_prob: keep_prob})
        _, dec_loss_val = sess.run([self.dec_solver, self.dec_loss], feed_dict={self.X: batch_xs, self.keep_prob: keep_prob})
        _, enc_loss_val = sess.run([self.enc_solver, self.enc_loss], feed_dict={self.X: batch_xs, self.keep_prob: keep_prob})
        total_loss_val = dis_loss_val + dec_loss_val + enc_loss_val

        if log_flag == True:
            self.logger.debug('Epoch %.3i, Batch[%.3i/%i], Enc loss : %.4E, Dec loss : %.4E, Dis loss : %.4E, Train loss: %.4E' % (epoch_idx, batch_idx + 1, batch_total, enc_loss_val, dec_loss_val, dis_loss_val, total_loss_val))
        return total_loss_val
        

    def inference(self, sess, batch_xs, epoch_idx, batch_idx, batch_total, log_flag, keep_prob):
        enc_loss_val, dec_loss_val, dis_loss_val = sess.run([self.enc_loss, self.dec_loss, self.dis_loss], feed_dict={self.X: batch_xs, self.keep_prob: keep_prob})
        total_loss_val = dis_loss_val + dec_loss_val + enc_loss_val

        if log_flag == True:
            self.logger.debug('Epoch %.3i, Batch[%.3i/%i], Enc loss : %.4E, Dec loss : %.4E, Dis loss : %.4E, Valid loss: %.4E' % (epoch_idx, batch_idx + 1, batch_total, enc_loss_val, dec_loss_val, dis_loss_val, total_loss_val))
        return total_loss_val

    def inference_with_top_k(self, sess, batch_xs, epoch_idx, batch_idx, batch_total, log_flag, keep_prob, k):
        enc_loss_val, dec_loss_val, dis_loss_val, top_k = sess.run([self.enc_loss, self.dec_loss, self.dis_loss, self.top_k_op], feed_dict={self.X: batch_xs, self.keep_prob: keep_prob, self.k: k})
        total_loss_val = dis_loss_val + dec_loss_val + enc_loss_val

        if log_flag == True:
            self.logger.debug('Epoch %.3i, Batch[%.3i/%i], Enc loss : %.4E, Dec loss : %.4E, Dis loss : %.4E, Test loss: %.4E' % (epoch_idx, batch_idx + 1, batch_total, enc_loss_val, dec_loss_val, dis_loss_val, total_loss_val))
        return total_loss_val, top_k

    def inference_with_recon(self, sess, batch_xs, epoch_idx, batch_idx, batch_total, log_flag, keep_prob):
        enc_loss_val, dec_loss_val, dis_loss_val, recon_val = sess.run([self.enc_loss, self.dec_loss, self.dis_loss, self.recon_X], feed_dict={self.X: batch_xs, self.keep_prob: keep_prob})
        total_loss_val = dis_loss_val + dec_loss_val + enc_loss_val

        if log_flag == True:
            self.logger.debug('Epoch %.3i, Batch[%.3i/%i], Enc loss : %.4E, Dec loss : %.4E, Dis loss : %.4E, Test loss: %.4E' % (epoch_idx, batch_idx + 1, batch_total, enc_loss_val, dec_loss_val, dis_loss_val, total_loss_val))
        return total_loss_val, recon_val
