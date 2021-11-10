import numpy as np


def data_categorization(data, label, PRI_jet_num):
    """
    This function is used to split data according to its PRI_jet_num
    The output is a dataset of one category i.e 0, 1, 2, 3

    @param data: features of the dataset
    @param label: labels of the dataset
    @param PRI_jet_num:
    @return: the features and labels of category PRI_jet_num
    """
    indexes = np.where(data[:, 22] == PRI_jet_num)
    features = data[indexes]
    labels = label[indexes]

    return features, labels


def data_sub_categorization(data, labels):
    """
    This function is used to sub categorize each category according to DER_mass_MMC

    @param data: features of the dataset
    @param labels: labels of the dataset
    @return: subset of data and labels where DER_mass_MMC is undefined, and subset of data and labels where DER_mass_MMC
    is defined
    """
    indexes = np.where(data[:, 0] == -999)
    undefined_mass_data = data[indexes]
    undefined_mass_labels = labels[indexes]

    indexes = np.where(data[:, 0] != -999)
    defined_mass_data = data[indexes]
    defined_mass_labels = labels[indexes]

    return undefined_mass_data, undefined_mass_labels, defined_mass_data, defined_mass_labels


def undefined_value_cleaning(data, column_names):
    """
    This functions deletes the columns where all values are undefined (== -999). It also updates column_names dropping
    the names of the deleted columns.

    @param data: features of the dataset
    @param column_names: feature column names
    @return: data without undefined features, and updated column_names
    """
    mask = np.all(data[..., :] == -999, axis=0)
    idx = np.argwhere(mask)
    features = np.delete(data, idx, axis=1)
    col_names = np.delete(column_names, idx)
    return features, col_names


def category_column_delete(data, PRI_jet_num, column_names):
    """
    This function deletes the columns containing values equal to the category number of out feature (PRI_jet_num).
    It also updates column_names dropping the names of the deleted columns.

    @param data: features of the category subset
    @param PRI_jet_num: category number
    @param column_names: feature column names
    @return: data without the category number column, and updated column_names
    """
    mask = np.all(data[..., :] == PRI_jet_num, axis=0)
    idx = np.argwhere(mask)
    features = np.delete(data, idx, axis=1)
    col_names = np.delete(column_names, idx)
    return features, col_names


def data_transformation(data, PRI_jet_num, column_names):
    """
    This function applies transformations to the features.
    The features to which the transformation is applied depends on the category PRI_jet_num.

    @param data: features of the category subset
    @param PRI_jet_num: category number
    @param column_names: feature column names
    @return: data with transformed feature values
    """
    new_data = data.copy()

    if PRI_jet_num == 0:
        columns_log = ['DER_mass_transverse_met_lep', 'DER_mass_vis', 'DER_pt_h', 'DER_pt_tot', 'DER_sum_pt',
                       'PRI_tau_pt', 'PRI_lep_pt', 'PRI_met']
    if PRI_jet_num == 1:
        columns_log = ['DER_mass_transverse_met_lep', 'PRI_met', 'PRI_met_sumet', 'DER_pt_tot', 'PRI_tau_pt',
                       'PRI_lep_pt', 'PRI_jet_leading_pt', 'PRI_jet_all_pt']
    if PRI_jet_num == 2:
        columns_log = ['DER_mass_transverse_met_lep', 'DER_pt_tot', 'PRI_tau_pt', 'PRI_met_sumet', 'PRI_lep_pt',
                       'PRI_met', 'PRI_jet_leading_pt', 'PRI_jet_subleading_pt', 'DER_pt_ratio_lep_tau', 'DER_pt_h',
                       'DER_mass_jet_jet']
        columns_sqrt = ['DER_deltaeta_jet_jet']
    if PRI_jet_num == 3:
        columns_log = ['DER_mass_transverse_met_lep', 'DER_mass_jet_jet', 'PRI_tau_pt', 'PRI_lep_pt', 'PRI_met',
                       'DER_mass_vis', 'DER_pt_tot', 'DER_pt_ratio_lep_tau']
        columns_sqrt = ['DER_deltaeta_jet_jet']

    # applying log on selected columns i.e columns_log
    for col in columns_log:
        if col in column_names:
            index = np.where(column_names == col)[0][0]
            new_data[:, index] = apply_log(new_data[:, index])

    # applying square root on selected columns i.e columns_sqrt (only in category 2 and 3)
    if PRI_jet_num == 2 or PRI_jet_num == 3:
        for col in columns_sqrt:
            if col in column_names:
                index = np.where(column_names == col)[0][0]
                new_data[:, index] = np.sqrt(new_data[:, index])

    return new_data


def apply_log(column):
    """
    This function apply the logarithm to a column
    If some entries are zero, it adds 1 to the feature before applying the log

    @param column: feature column
    @return: transformed feature column
    """
    new_column = column.copy()
    if 0 in new_column:
        new_column = new_column + 1
    new_column = np.log(new_column)
    return new_column


def delete_correlated_features(tx, threshold, column_names):
    """
    This function computes the correlation between the columns of tX. Then the column with the smallest index is removed
    for every two correlated columns. It also updates the column names after deleting some features.

    @param tx: dataset
    @param threshold: correlation threshold
    @param column_names: feature names
    @return: uncorrelated dataset, and updated column names
    """
    corr_m = np.corrcoef(tx, rowvar=False)
    indices_one, indices_two = np.where(np.abs(np.tril(corr_m, -1)) >= threshold)
    removed_cols = list(set(indices_two))
    return np.delete(tx, removed_cols, axis=1), np.delete(column_names, removed_cols)


def data_augmentation(data, max_degree):
    """
    This function augments our features by computing their powers from 1 to max_degree

    @param data: dataset
    @param max_degree: maximum degree
    @return: dataset with number of features * max_degree columns
    """
    augmented = np.c_[np.ones((data.shape[0], 1)), data.copy()]
    for i in range(1, max_degree):
        augmented = np.c_[augmented, data ** (i + 1)]

    return augmented


def standardize_data(data):
    """
    This function standardizes data and returns the mean and std

    @param data: dataset
    @return: standardized dataset, mean of dataset and its std
    """
    mean = np.mean(data, axis=0)
    standard_data = data - mean
    std = np.std(standard_data, axis=0)
    standard_data = standard_data / std
    return standard_data, mean, std


def delete_columns_from_list(data, column_names, del_col):
    """
    This function deletes the columns in del_col from dataset according to the index of each column name in column_names

    @param data: dataset
    @param column_names: dataset column names
    @param del_col: names of the columns to delete
    @return: dataset without the columns in del_col, and its new column names
    """
    indexes = np.where(np.in1d(column_names, del_col))[0]
    c_names = np.delete(column_names, indexes)
    features = np.delete(data, indexes, axis=1)
    return features, c_names
