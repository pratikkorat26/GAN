import tensorflow as tf
import tensorlayer as tl
import time
import os
from srgan.dataset import get_dataset
from srgan.models import generator , discriminator
from srgan.config import config

def train(generator , discriminator , config):

    VGG = tf.keras.applications.vgg19.VGG19(include_top = False)

    lr_v = tf.Variable(config.TRAIN.lr_v)

    g_optimizer_init = tf.optimizers.Adam(lr_v, beta_1 = config.TRAIN.beta1)
    g_optimizer = tf.optimizers.Adam(lr_v, beta_1 = config.TRAIN.beta1)
    d_optimizer = tf.optimizers.Adam(lr_v, beta_1 = config.TRAIN.beta1)

    # generator.train()
    # discriminator.train()
    VGG.trainable = False

    print(f"loading data")
    train_ds = get_dataset(config.TRAIN.img_dir) #==> Training data


    ## initialize learning (G)
    print(f"started initialized training")
    # n_step_epoch = round(config.TRAIN.n_epoch_init // config.TRAIN.batch_size)
    # for epoch in range(config.TRAIN.n_epoch_init):
    #     for step, (hr_img, lr_img) in enumerate(train_ds):
    #         if lr_img.shape[0] != config.TRAIN.batch_size: # if the remaining data in this epoch < batch_size
    #             break
    #
    #         step_time = time.time()
    #         with tf.GradientTape() as tape:
    #             fake_hr_patchs = generator(lr_img)
    #             mse_loss = tl.cost.mean_squared_error(fake_hr_patchs, hr_img, is_mean=True)
    #
    #         grad = tape.gradient(mse_loss, generator.trainable_weights)
    #         g_optimizer_init.apply_gradients(zip(grad, generator.trainable_weights))
    #
    #         print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, mse: {:.3f} ".format(
    #             epoch, config.TRAIN.n_epoch_init, step, n_step_epoch, time.time() - step_time, mse_loss))
    #     if (epoch != 0) and (epoch % 10 == 0):
    #         tl.vis.save_images(fake_hr_patchs.numpy(), [2, 4], os.path.join(config.save_dir, 'train_g_init_{}.png'.format(epoch)))

    print(f"hello adverserial training")
    ## adversarial learning (G, D)
    n_step_epoch = round(config.TRAIN.n_epoch_init // config.TRAIN.batch_size)
    for epoch in range(config.TRAIN.n_epoch):
        for step, (hr_img, lr_img) in enumerate(train_ds):
            if lr_img.shape[0] != config.TRAIN.batch_size: # if the remaining data in this epoch < batch_size
                break
            step_time = time.time()
            with tf.GradientTape(persistent=True) as tape:
                fake_patchs = generator(lr_img)
                logits_fake = discriminator(fake_patchs)
                logits_real = discriminator(hr_img)
                feature_fake = VGG((fake_patchs+1)/2.) # the pre-trained VGG uses the input range of [0, 1]
                feature_real = VGG((hr_img+1)/2.)

                #Losses of given generator and discriminator
                d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real))
                d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake))
                d_loss = 100*(d_loss1 + d_loss2)
                g_gan_loss = 1e-2 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake))
                mse_loss = tl.cost.mean_squared_error(fake_patchs, hr_img, is_mean=True)
                vgg_loss = 2e-3 * tl.cost.mean_squared_error(feature_fake, feature_real, is_mean=True)
                g_loss = mse_loss + vgg_loss + g_gan_loss
            print(f"vgg loss : {vgg_loss}" )
            grad = tape.gradient(g_loss, generator.trainable_weights)
            g_optimizer.apply_gradients(zip(grad, generator.trainable_weights))
            grad = tape.gradient(d_loss, discriminator.trainable_weights)
            d_optimizer.apply_gradients(zip(grad, discriminator.trainable_weights))
            print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, g_loss(mse:{:.3f}, vgg:{:.3f}, adv:{:.3f}) d_loss: {:.3f}".format(
                epoch, config.TRAIN.n_epoch_init, step, n_step_epoch, time.time() - step_time, mse_loss, vgg_loss, g_gan_loss, d_loss))

        # update the learning rate
        if epoch != 0 and (epoch % config.decay_every == 0):
            new_lr_decay = config.lr_decay**(epoch // config.decay_every)
            lr_v.assign(config.lr_init * new_lr_decay)
            log = " ** new learning rate: %f (for GAN)" % (config.lr_init * new_lr_decay)
            print(log)

        if (epoch != 0) and (epoch % 10 == 0):
            tl.vis.save_images(fake_patchs.numpy(), [2, 4], os.path.join(config.save_dir, 'train_g_{}.png'.format(epoch)))
            generator.save_weights(os.path.join(config.checkpoint_dir, 'g.h5'))
            discriminator.save_weights(os.path.join(config.checkpoint_dir, 'd.h5'))

if __name__ == "__main__":
    gen = generator((64, 64, 3), 1)
    disc = discriminator((256,256,3) , 1)

    train(gen , disc , config)

