import tensorflow as tf
import os
import time
import numpy as np
import json
from datetime import datetime
import argparse


from helpers import TDW_data
from model import ResNet18, ProjectionHead
from helpers import byol_loss
from pathlib import Path
#import tensorflow_addons as tfa


def exp_moving_average(online_f, online_g, target_f, target_g, beta):
    target_f_weights = target_f.get_weights()
    online_f_weights = online_f.get_weights()
    target_g_weights = target_g.get_weights()
    online_g_weights = online_g.get_weights()
    for i in range(len(online_f_weights)):
        target_f_weights[i] = beta * target_f_weights[i] + (1 - beta) * online_f_weights[i]
    for i in range(len(online_g_weights)):
        target_g_weights[i] = beta * target_g_weights[i] + (1 - beta) * online_g_weights[i]
    target_f.set_weights(target_f_weights)
    target_g.set_weights(target_g_weights)

@tf.function
def train_step_pretraining(opt, x1, x2, f_target, g_target, f_online, g_online, q_online): 
    h_target_1 = f_target(x1, training=True)
    z_target_1 = g_target(h_target_1, training=True)

    h_target_2 = f_target(x2, training=True)
    z_target_2 = g_target(h_target_2, training=True)

    with tf.GradientTape(persistent=True) as tape:
        h_online_1 = f_online(x1, training=True)
        z_online_1 = g_online(h_online_1, training=True)
        p_online_1 = q_online(z_online_1, training=True)
        
        h_online_2 = f_online(x2, training=True)
        z_online_2 = g_online(h_online_2, training=True)
        p_online_2 = q_online(z_online_2, training=True)
        
        p_online = tf.concat([p_online_1, p_online_2], axis=0)
        z_target = tf.concat([z_target_2, z_target_1], axis=0)
        loss = byol_loss(p_online, z_target)
    grads = tape.gradient(loss, f_online.trainable_variables)
    opt.apply_gradients(zip(grads, f_online.trainable_variables))
    grads = tape.gradient(loss, g_online.trainable_variables)
    opt.apply_gradients(zip(grads, g_online.trainable_variables))
    grads = tape.gradient(loss, q_online.trainable_variables)
    opt.apply_gradients(zip(grads, q_online.trainable_variables))
    del tape
    return loss

def main_training(args):
    folder_name = datetime.now().strftime("%d%m%Y_%H%M%S")
    os.mkdir(folder_name)
    os.mkdir(f"{folder_name}/weights")
    os.mkdir(f"{folder_name}/model")

    # Load data
    data = TDW_data(r"C:\Users\felix\Documents\Master\Thesis\tdw\photoreal_80x80")
    
    # Define optimizer
    lr = args.learning_rate
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    #opt = tfa.optimizers.LAMB(learning_rate = lr)
    
    batches_per_epoch = data.num_train_images // args.batch_size
    log_every = 10  # batches
    save_every = args.save_every # epochs
    #beta_base = args.beta
    beta = args.beta

    with open(f"{folder_name}/config.json","w") as fp:
         config = vars(args)
         config["optimizer"] = opt.get_config()
         json.dump(config, fp)

    # Instantiate networks
    f_online = ResNet18()
    g_online = ProjectionHead()
    q_online = ProjectionHead()
    f_target = ResNet18()
    g_target = ProjectionHead()

    # Initialize the weights of the networks
    x = tf.random.normal((200, 80, 80, 3))
    h = f_online(x, training=False)
    f_online.save(f"{folder_name}/model")
    z = g_online(h, training=False)
    p = q_online(z, training=False)
    h = f_target(x, training=False)
    z = g_target(h, training=False)
  
    losses = []
   
    for epoch_id in range(args.num_epochs):
        for batch_id in range(batches_per_epoch):
            x1, x2 = data.get_batch_training(batch_id, args.batch_size)
            loss = train_step_pretraining(opt, x1, x2, f_target, g_target, f_online, g_online, q_online)
            losses.append(float(loss))
            #beta = 1-(1-beta_base)*(np.cos(np.pi*epoch_id/args.num_epochs)+1)/2
            exp_moving_average(f_online, g_online, f_target, g_target, beta)

            if (batch_id + 1) % log_every == 0:
                print('[Epoch {}/{} Batch {}/{}] Loss={:.5f}.'.format(epoch_id+1, args.num_epochs, batch_id+1, batches_per_epoch, loss))

        if (epoch_id + 1) % save_every == 0:
            f_online.save_weights(f'{folder_name}/weights/f_online_{epoch_id+1:03}.h5')
            print('Weights of f saved.')
            print("Losses:",tf.stack(losses).numpy())
            
    np.savetxt(f'{folder_name}/losses.txt', tf.stack(losses).numpy())
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size for pretraining')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.99, help='Exponential Moving average')
    parser.add_argument('--save_every', type=int, default=10, help='How often weights should be stored')
    parser.add_argument('--description', type=str, default="-", help='Optional: Short description about the experiment...')


    args = parser.parse_args()
    main_training(args)