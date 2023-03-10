# -*- coding: utf-8 -*-
import keras.backend as K
from keras.losses import binary_crossentropy
from keras.losses import categorical_crossentropy
from keras.utils.generic_utils import get_custom_objects

from metrics import jaccard_score, f_score

SMOOTH = 1e-12

__all__ = [
    'jaccard_loss', 'bce_jaccard_loss', 'cce_jaccard_loss',
    'dice_loss', 'bce_dice_loss', 'cce_dice_loss','dice_coef_loss'
]

# ============================== Jaccard Losses ==============================

def jaccard_loss(gt, pr, class_weights=1., smooth=SMOOTH, per_image=True):
    return 1 - jaccard_score(gt, pr, class_weights=class_weights, smooth=smooth, per_image=per_image)


def bce_jaccard_loss(gt, pr, bce_weight=1., smooth=SMOOTH, per_image=True):
    bce = K.mean(binary_crossentropy(gt, pr))
    loss = bce_weight * bce + jaccard_loss(gt, pr, smooth=smooth, per_image=per_image)
    return loss


def cce_jaccard_loss(gt, pr, cce_weight=1., class_weights=1., smooth=SMOOTH, per_image=True):
    cce = categorical_crossentropy(gt, pr) * class_weights
    cce = K.mean(cce)
    return cce_weight * cce + jaccard_loss(gt, pr, smooth=smooth, class_weights=class_weights, per_image=per_image)


get_custom_objects().update({
    'jaccard_loss': jaccard_loss,
    'bce_jaccard_loss': bce_jaccard_loss,
    'cce_jaccard_loss': cce_jaccard_loss,
})


# ============================== Dice Losses ================================

def dice_loss(gt, pr, class_weights=1., smooth=SMOOTH, per_image=True):
    return 1 - f_score(gt, pr, class_weights=class_weights, smooth=smooth, per_image=per_image, beta=1.)


def bce_dice_loss(gt, pr, bce_weight=1., smooth=SMOOTH, per_image=True):
    bce = K.mean(binary_crossentropy(gt, pr))
    loss = bce_weight * bce + dice_loss(gt, pr, smooth=smooth, per_image=per_image)
    return loss


def cce_dice_loss(gt, pr, cce_weight=1., class_weights=1., smooth=SMOOTH, per_image=True):
    cce = categorical_crossentropy(gt, pr) * class_weights
    cce = K.mean(cce)
    return cce_weight * cce + dice_loss(gt, pr, smooth=smooth, class_weights=class_weights, per_image=per_image)

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

get_custom_objects().update({
    'dice_loss': dice_loss,
    'bce_dice_loss': bce_dice_loss,
    'cce_dice_loss': cce_dice_loss,
    'dice_coef_loss':dice_coef_loss,
})