import numpy as np

import warnings
warnings.filterwarnings('ignore')

#### Tensorflow imports ###
import tensorflow as tf
import tensorflow.keras.backend as K

def concordance_loss(ground_truth, prediction):
    """Defines concordance loss for training the model. 

    Args:
       prediction: prediction of the model.
       ground_truth: ground truth values.
    Returns:
       The concordance value.
    """

    pred_mean, pred_var = K.mean(prediction), K.var(prediction)
    gt_mean, gt_var = K.mean(ground_truth), K.var(ground_truth)
    covariance = K.mean((prediction-pred_mean)*(ground_truth-gt_mean))
    denominator = (gt_var + pred_var + (gt_mean - pred_mean) ** 2)

    concordance_cc2 = (2 * covariance) / denominator

    #valor normalizado entre 0 e 1
    #concordance_cc2 = (concordance_cc2 + 1) / 2

    return 1-concordance_cc2

def concordance_cc(prediction, ground_truth):
    """Concordance correlation coefficient.

    The concordance correlation coefficient is a measure of inter-rater agreement.
    It measures the deviation of the relationship between predicted and true values
    from the 45 degree angle.
    Read more: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    Original paper: Lawrence, I., and Kuei Lin. "A concordance correlation coefficient to evaluate reproducibility." Biometrics (1989): 255-268.  
    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    Returns
    -------
    loss : A float in the range [-1,1]. A value of 1 indicates perfect agreement
    between the true and the predicted values.
    Examples
    --------
    >>> from sklearn.metrics import concordance_correlation_coefficient
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> concordance_correlation_coefficient(y_true, y_pred)
    0.97678916827853024
    """

    #CCC = 2 * COVAR[X,Y] / (VAR[X] + VAR[Y] + (E[X] - E[Y])^2)

    # Mean, Variance, Standard deviation
    pred_mean, pred_var, pred_sd = np.mean(prediction), np.var(prediction), np.std(prediction)
    gt_mean, gt_var, gt_sd = np.mean(ground_truth), np.var(ground_truth), np.var(ground_truth)

    # Calculate CCC LUCIANA
    covariance = np.mean((prediction-pred_mean)*(ground_truth-gt_mean))
    denominator = (gt_var + pred_var + (gt_mean - pred_mean) ** 2)
    concordance_cc2 = (2 * covariance) / denominator

    # Calculate CCC https://gitlab.com/-/snippets/1730605
    #covariance = np.cov([ground_truth, prediction])[0,1]
    #denominator = (gt_var + pred_var + (gt_mean - pred_mean) ** 2)
    #concordance_cc2 = (2 * covariance) / denominator

    return concordance_cc2
