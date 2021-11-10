import numpy as np
from proj1_helpers import predict_labels
from implementations import ridge_regression, mse_loss


def build_n_indices(y, n_fold):
    """
    build n subsets of the dataset, to be used as test sets for cross validation
    @param y: np.ndarray, array of shape (num_samples,) representing the true labels of the data
    @param n_fold: number of subsets
    @return: nd.array containing the indices corresponding to each subset
    """
    num_row = y.shape[0]
    interval = int(num_row / n_fold)
    indices = np.random.permutation(num_row)
    n_indices = [indices[n * interval: (n + 1) * interval]
                 for n in range(n_fold)]
    return np.array(n_indices)


def cross_validate(y, tx, train, k, n, loss_f, seed=0):
    """
    perform k * n cross validation, testing accuracy, f1-score and loss
    @param y: np.ndarray, array of shape (num_samples,) representing the true labels of the data
    @param tx: np.ndarray, array of shape (num_samples, num_features) representing the features
    @param train: function that should perform the training, should take y and tx as parameters
                  and return the weights and loss
    @param k: int, number of times to perform the splitting operation
    @param n: int, number of subsets to split the data into
    @param loss_f: the loss function of the model, should take as parameters y and tx and return a single float
    @param seed: int, randomizing seed
    @return: 3 np.ndarrays of shape (k*n,), respectively containing the accuracy, f1-score and loss in that order
    """
    np.random.seed(seed)
    # tables to store the metrics
    accuracy_table = []
    f1_score_table = []
    loss_table = []

    for i in range(k):
        # build the n test subsets
        test_indices = build_n_indices(y, n)
        for test_set in test_indices:
            # compute the train test
            train_set = ~np.in1d(range(y.shape[0]), test_set)
            # split the data into test and train set
            x_test, y_test = tx[test_set], y[test_set]
            x_train, y_train = tx[train_set], y[train_set]
            # compute the mean and std of the features, column-wise
            x_train_mean, x_train_std = np.mean(x_train, axis=0), np.std(x_train, axis=0)
            # standardize the test set
            x_test = (x_test - x_train_mean) / x_train_std
            # train the model on the train set
            w, loss = train(y_train, (x_train - x_train_mean) / x_train_std)
            # compute the predictions on the test set
            pred_test = predict_labels(w, x_test)
            # compute the metrics
            tps = np.count_nonzero(np.logical_and(pred_test == y_test, pred_test == 1))
            fps = np.count_nonzero(np.logical_and(pred_test != y_test, pred_test == 1))
            fns = np.count_nonzero(np.logical_and(pred_test != y_test, pred_test == -1))
            accuracy = np.count_nonzero(pred_test == y_test) / y_test.shape[0]
            f1_score = tps / (tps + 0.5 * (fps + fns))
            # store the metrics
            f1_score_table.append(f1_score)
            accuracy_table.append(accuracy)
            loss_table.append(loss_f(y_test, x_test, w))

    return np.array(accuracy_table), np.array(f1_score_table), np.array(loss_table)


def find_best_param_ridge_reg(y, tx, max_polinom, lambda_min, lambda_max, num_points, seed=0):
    """
    Performs cross-validation to find the best parameters for ridge regression, i.e. the polynomial degree
    for the data augmentation and the regularization rate lambda. Uses accuracy to choose the best parameters.
    @param y: np.ndarray, array of shape (num_samples,) representing the true labels of the data
    @param tx: np.ndarray, array of shape (num_samples, num_features) representing the features
    @param max_polinom: int, maximum polynomial degree to augment the data to
    @param lambda_min: float, minimum lambda value to start the search from (log scale)
    @param lambda_max: float, maximum lambda value to stop the search at (included in the search space) (log scale)
    @param num_points: int, number of points to consider for lambda
    @param seed: intm randomizing seed
    @return: best polynomial degree, best lambda_, loss, accuracy, f1-score
    """
    # initialize the search space for lambda, using a log scale
    lambda_space = np.logspace(lambda_min, lambda_max, num=num_points, endpoint=True, base=10.0)
    # initialize matrices to store our metrics
    accuracy_matrix = np.zeros((max_polinom, num_points))
    f1_score_matrix = np.zeros((max_polinom, num_points))
    loss_matrix = np.zeros((max_polinom, num_points))
    # for more efficient data augmentation, we do it iteratively and store the previous result each time
    tx_augmented = tx
    tx_temp = tx
    for degree in range(1, max_polinom + 1):
        for index, lambda_ in enumerate(lambda_space):
            # create the training function
            ridge_reg_train = lambda y, tx: ridge_regression(y, tx, lambda_)
            # perform 2 x 10 cross-validation
            accuracy_table, f1_score_table, loss_table = cross_validate(y, tx_augmented, ridge_reg_train, \
                                                                        k=2, n=10, loss_f=mse_loss)
            # store the mean value of each metric in the corresponding matrix
            accuracy_matrix[degree - 1, index] = np.mean(accuracy_table)
            f1_score_matrix[degree - 1, index] = np.mean(f1_score_table)
            loss_matrix[degree - 1, index] = np.mean(loss_table)

        tx_temp = tx_temp * tx
        tx_augmented = np.c_[tx_augmented, tx_temp]

    # find the best parameters corresponding to the best accuracy

    min_loss = np.unravel_index(np.argmin(loss_matrix), loss_matrix.shape)
    loss = loss_matrix[min_loss]
    accuracy = accuracy_matrix[min_loss]
    f1_score = f1_score_matrix[min_loss]
    best_polinom = min_loss[0] + 1
    best_lambda_ = lambda_space[min_loss[1]]

    return best_polinom, best_lambda_, loss, accuracy, f1_score
