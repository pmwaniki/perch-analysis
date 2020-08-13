import tensorflow as tf
import numpy as np

def compute_loss_perch(model,perch_x,rev_x,perch_y):
    pred_perch=model.model_perch([perch_x,rev_x])
    loss_perch=tf.keras.losses.categorical_crossentropy(perch_y,pred_perch)
    loss_perch=tf.reduce_mean(loss_perch)+tf.add_n(model.model_perch.losses)
    return loss_perch

def compute_loss_chestray(model,chestray_x,chestray_y):
    pred_chestray=model.model_chestray(chestray_x)
    loss_chestray=tf.keras.losses.binary_crossentropy(chestray_y,pred_chestray)
    loss_chestray=tf.reduce_mean(loss_chestray)+tf.add_n(model.model_chestray.losses)
    return loss_chestray

def compute_gradients_perch(model,perch_x,rev_x,perch_y):
    with tf.GradientTape() as tape:
        loss_perch=compute_loss_perch(model,perch_x,rev_x,perch_y)
        gradient_perch=tape.gradient(loss_perch,model.model_perch.trainable_variables)
    return gradient_perch,loss_perch

def compute_gradients_chestray(model, chestray_x, chestray_y):
    with tf.GradientTape() as tape:
        loss_chestray=compute_loss_chestray(model,chestray_x,chestray_y)
        gradient_chestray=tape.gradient(loss_chestray,model.model_chestray.trainable_variables)
    return gradient_chestray,loss_chestray

def apply_gradients_perch(model,optimizer_perch,gradients):
    optimizer_perch.apply_gradients(zip(gradients,model.model_perch.trainable_variables))

def apply_gradients_chestray(model,optimizer_chestray,gradients):
    optimizer_chestray.apply_gradients(zip(gradients,model.model_chestray.trainable_variables))

# @tf.function
def train_one_batch(model,perch_x,rev_x,perch_y,chestray_x,chestray_y,norm,optimizer_perch,optimizer_chestray):
    def start_perch():
        gradients_perch, losses_perch = compute_gradients_perch(model, perch_x, rev_x, perch_y)
        # norm_gradients_perch = [t * norm / tf.norm(t) for t in gradients_perch]
        norm_gradients_perch = [tf.clip_by_norm(t, norm) for t in gradients_perch]
        apply_gradients_perch(model, optimizer_perch, norm_gradients_perch)
        gradients_chestray, losses_chestray = compute_gradients_chestray(model, chestray_x, chestray_y)
        # norm_gradients_chestray = [t * norm / tf.norm(t) for t in gradients_chestray]
        norm_gradients_chestray = [tf.clip_by_norm(t, norm) for t in gradients_chestray]
        apply_gradients_chestray(model, optimizer_chestray, norm_gradients_chestray)
        return gradients_perch, gradients_chestray, losses_perch, losses_chestray
    def start_chestray():
        gradients_chestray, losses_chestray = compute_gradients_chestray(model, chestray_x, chestray_y)
        # norm_gradients_chestray = [t * norm / tf.norm(t) for t in gradients_chestray]
        norm_gradients_chestray = [tf.clip_by_norm(t, norm) for t in gradients_chestray]
        apply_gradients_chestray(model, optimizer_chestray, norm_gradients_chestray)
        gradients_perch, losses_perch = compute_gradients_perch(model, perch_x, rev_x, perch_y)
        # norm_gradients_perch = [t * norm / tf.norm(t) for t in gradients_perch]
        norm_gradients_perch = [tf.clip_by_norm(t, norm) for t in gradients_perch]
        apply_gradients_perch(model, optimizer_perch, norm_gradients_perch)
        return gradients_perch, gradients_chestray, losses_perch, losses_chestray
    return tf.cond(tf.random.uniform([], 0, 1)>0.5,start_perch,start_chestray)
    # return start_chestray()
