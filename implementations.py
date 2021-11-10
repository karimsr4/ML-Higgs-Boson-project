import numpy as np


def gradient_descent(y, tx, initial_w, max_iters, gamma, gradient_f, loss_f):
    """
    perform gradient descent to find the minimum of the loss function

    @param y: np.ndarray, array of shape (num_samples,) representing the true labels of the data
    @param tx: np.ndarray, array of shape (num_samples, num_features) representing the features
    @param initial_w: np.ndarray, array of shape (num_features,) representing the initial weights of the model
    @param max_iters: int, maximum number of iterations of the gradient descent
    @param gamma: float, learning rate
    @param gradient_f: function taking y,tx,w and returning the gradient of the loss function
    @param loss_f: loss function of the model
    @return: best weights, corresponding loss
    """
    w = initial_w
    for n_iter in range(max_iters):
        w = w - gamma * gradient_f(y, tx, w)
    return w, loss_f(y, tx, w)


def stoch_gradient_descent(y, tx, initial_w, max_iters, gamma, gradient_f, loss_f):
    """
    perform stochastic gradient descent to find the minimum of the loss function
    @param y: np.ndarray, array of shape (num_samples,) representing the true labels of the data
    @param tx: np.ndarray, array of shape (num_samples, num_features) representing the features
    @param initial_w: np.ndarray, array of shape (num_features,) representing the initial weights of the model
    @param max_iters: int, maximum number of iterations of the gradient descent
    @param gamma: float, learning rate
    @param gradient_f: function taking y,tx,w and returning the gradient of the loss function
    @param loss_f: loss function of the model
    @return: best weights, corresponding loss
    """
    w = initial_w
    for n_iter in range(max_iters):
        sample = np.random.choice(tx.shape[0])
        # make the sample a 2D array, as the gradient function does not work on 1D np.ndarray
        sample_x, sample_y = tx[sample][np.newaxis, ...], y[sample]
        w = w - gamma * gradient_f(sample_y, sample_x, w)
    return w, loss_f(y, tx, w)


def mse_gradient(y, tx, w):
    """
    calculate the gradient of the MSE loss function
    @param y: np.ndarray, array of shape (num_samples,) representing the true labels of the data
    @param tx: np.ndarray, array of shape (num_samples, num_features) representing the features tx
    @param w: np.ndarray, array of shape (num_features,) representing the weights of the linear model
    @return: np.ndarray of shape (num_features,) representing the gradient of the MSE loss function
    """
    e = y - tx.dot(w)
    n = tx.shape[0]
    gradient = (- 1 / n) * (tx.T.dot(e))
    return gradient


def mse_loss(y, tx, w):
    """
    compute the MSE loss of the model
    @param y: np.ndarray, array of shape (num_samples,) representing the true labels of the data
    @param tx: np.ndarray, array of shape (num_samples, num_features) representing the features tx
    @param w: np.ndarray, array of shape (num_features,) representing the weights of the linear model
    @return: float, the MSE loss of the model
    """
    e = y - (tx.dot(w))
    n = tx.shape[0]
    loss = (e * e).sum()
    return (1 / (2 * n)) * loss


def sigmoid(t):
    """
    compute the sigmoid of the parameter t
    @param t: np.ndarray or float
    @return: sigmoid function applied to the inpute
    """
    return 1 / (1 + np.exp(-t))


def log_loss_gradient(y, tx, w):
    """
    Compute gradient of the negative log likelihood function
    @param y: np.ndarray, array of shape (num_samples,) representing the true labels of the data
    @param tx: np.ndarray, array of shape (num_samples, num_features) representing the features tx
    @param w: np.ndarray, array of shape (num_features,) representing the weights of the logistic model
    @return: np.ndarray of shape (num_features,) representing the gradient of the negative log likelihood
    """
    return tx.T.dot(sigmoid(tx.dot(w)) - y)


def neg_log_loss(y, tx, w):
    """
    Compute the negative log likelihood of the function
    @param y: np.ndarray, array of shape (num_samples,) representing the true labels of the data
    @param tx: np.ndarray, array of shape (num_samples, num_features) representing the features tx
    @param w: np.ndarray, array of shape (num_features,) representing the weights of the logistic model
    @return: float, the negative log likelihood
    """
    data_dot_w = np.dot(tx, w)
    # use np.logaddexp as it can deal with overflows
    log_loss_v = np.logaddexp(0, data_dot_w) - y * data_dot_w
    return log_loss_v.sum()


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    perform gradient descent to find the minimum of the MSE loss function
    @param y: np.ndarray, array of shape (num_samples,) representing the true labels of the data
    @param tx: np.ndarray, array of shape (num_samples, num_features) representing the features
    @param initial_w: np.ndarray, array of shape (num_features,) representing the initial weights of the model
    @param max_iters: int, maximum number of iterations of the gradient descent
    @param gamma: float, learning rate
    @return: best weights, corresponding loss
    """
    return gradient_descent(y, tx, initial_w, max_iters, gamma, mse_gradient, mse_loss)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    perform stochastic gradient descent to find the minimum of the MSE loss function
    @param y: np.ndarray, array of shape (num_samples,) representing the true labels of the data
    @param tx: np.ndarray, array of shape (num_samples, num_features) representing the features
    @param initial_w: np.ndarray, array of shape (num_features,) representing the initial weights of the model
    @param max_iters: int, maximum number of iterations of the gradient descent
    @param gamma: float, learning rate
    @return: best weights, corresponding loss
    """
    return stoch_gradient_descent(y, tx, initial_w, max_iters, gamma, mse_gradient, mse_loss)


def least_squares(y, tx):
    """
    Find the closed form solution of the MSE loss function
    @param y: np.ndarray, array of shape (num_samples,) representing the true labels of the data
    @param tx: np.ndarray, array of shape (num_samples, num_features) representing the features
    @return: best weights, corresponding loss
    """
    tx_transpose = tx.T
    rhs = tx_transpose @ y
    lhs = tx_transpose @ tx
    w_star = np.linalg.solve(lhs, rhs)
    return w_star, mse_loss(y, tx, w_star)


def ridge_regression(y, tx, lambda_):
    """
    Find the closed form solution of the L2-regularized MSE loss function
    @param y: np.ndarray, array of shape (num_samples,) representing the true labels of the data
    @param tx: np.ndarray, array of shape (num_samples, num_features) representing the features
    @return: best weights, corresponding regularized loss
    """
    n = tx.shape[0]
    data_dim = tx.shape[1]
    tx_T = tx.T
    rhs = tx_T @ y
    # add the regularization term to the equation
    lhs = tx_T @ tx + (2 * n * lambda_) * np.identity(data_dim)
    w_star = np.linalg.solve(lhs, rhs)

    return w_star, mse_loss(y, tx, w_star)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    perform gradient descent to find the minimum of the negative log likelihodd loss function
    @param y: np.ndarray, array of shape (num_samples,) representing the true labels of the data
    @param tx: np.ndarray, array of shape (num_samples, num_features) representing the features
    @param initial_w: np.ndarray, array of shape (num_features,) representing the initial weights of the model
    @param max_iters: int, maximum number of iterations of the gradient descent
    @param gamma: float, learning rate
    @return: best weights, corresponding loss (negative log likelihood)
    """
    # we can map the -1's in the labels to 0's, making our formulas more compact
    y_c = y.copy()
    y_c[y_c == -1] = 0
    return gradient_descent(y_c, tx, initial_w, max_iters, gamma, log_loss_gradient, neg_log_loss)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    perform gradient descent to find the minimum of the L2-regularized negative log likelihodd loss function
    @param y: np.ndarray, array of shape (num_samples,) representing the true labels of the data
    @param tx: np.ndarray, array of shape (num_samples, num_features) representing the features
    @param initial_w: np.ndarray, array of shape (num_features,) representing the initial weights of the model
    @param max_iters: int, maximum number of iterations of the gradient descent
    @param gamma: float, learning rate
    @return: best weights, corresponding regularized loss (negative log likelihood)
    """

    # Adapt the gradient to add the regularization terms
    def reg_logistic_regression_gradient(y_r, tx_r, w):
        return log_loss_gradient(y_r, tx_r, w) + lambda_ * w

    # we can map the -1's in the labels to 0's, making our formulas more compact
    y_c = y.copy()
    y_c[y_c == -1] = 0
    return gradient_descent(y_c, tx, initial_w, max_iters, gamma, reg_logistic_regression_gradient,
                            neg_log_loss)
