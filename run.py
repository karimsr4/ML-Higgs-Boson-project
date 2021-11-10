# This is the main script to be run to train the model on the training set and test on test set

# necessary importations
import os
from data_manipulation import *
from proj1_helpers import *
from implementations import *

# path to the dataset files
DATA_TRAIN_PATH = "data" + os.path.sep + "train.csv"
DATA_TEST_PATH = "data" + os.path.sep + "test.csv"

#######################################################################################################################
# Data cleaning, feature engineering and training
#######################################################################################################################

# loading the train dataset
train_labels, train_set, ids = load_csv_data(DATA_TRAIN_PATH)

# Initialize lists that will contain weights, means, standard deviations, and deleted columns in data cleaning
weights_per_category = []
mean_per_category = []
std_per_category = []
deleted_columns = []

# constant value that indicate the names of columns, useful to keep track of columns of each category
COLUMN_NAMES = np.array(
    ['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis', 'DER_pt_h', 'DER_deltaeta_jet_jet',
     'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 'DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt',
     'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality', 'DER_lep_eta_centrality', 'PRI_tau_pt', 'PRI_tau_eta',
     'PRI_tau_phi', 'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi', 'PRI_met_sumet',
     'PRI_jet_num', 'PRI_jet_leading_pt', 'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt',
     'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_jet_all_pt'])

# number of categories
CATEGORIES_NBR = 4

# Best hyper-parameters selected after cross validation
with open('best_degrees.npy', 'rb') as f:
    best_degrees = np.load(f)

with open('best_lambdas.npy', 'rb') as f:
    best_lambdas = np.load(f)

# iterating over each jet_num
for jet_num in range(CATEGORIES_NBR):
    # For each jet, we isolate the rows of this category
    features, labels = data_categorization(train_set, train_labels, jet_num)

    # for each jet, we sub categorize the rows using the first feature i.e the mass_MMC feature
    mass_undefined_data, mass_undefined_labels, mass_defined_data, mass_defined_labels = data_sub_categorization(
        features, labels)

    retained_columns_def = COLUMN_NAMES.copy()
    retained_columns_undef = COLUMN_NAMES.copy()

    # For each sub category, we remove the columns that have unique value -999 i.e undefined, as these columns carry no
    # information
    # We also keep track of the columns retained after this first step of cleaning
    mass_undefined_data, retained_columns_undef = undefined_value_cleaning(mass_undefined_data, retained_columns_undef)

    mass_defined_data, retained_columns_def = undefined_value_cleaning(mass_defined_data, retained_columns_def)

    # For each sub category, we remove the columns that have unique value jet_num i.e the columns PRI_jet_num.
    # This also removes  some columns with unique values as PRI_jet_num. For example, when PRI_jet_num=0, PRI_jet_all_pt
    # is always 0, so will be removed too
    mass_undefined_data, retained_columns_undef = category_column_delete(mass_undefined_data, jet_num,
                                                                         retained_columns_undef)
    mass_defined_data, retained_columns_def = category_column_delete(mass_defined_data, jet_num, retained_columns_def)

    # For each sub category, we apply some transformations on the features, mainly applying logarithm function on right
    # skewed distributions and square root on 'DER_deltaeta_jet_jet', This choice was done after visualizing
    # distributions for each category and selecting the columns to apply log
    mass_undefined_data = data_transformation(mass_undefined_data, jet_num, retained_columns_undef)
    mass_defined_data = data_transformation(mass_defined_data, jet_num, retained_columns_def)

    # For each sub category, we determine highly correlated features and remove unneeded features
    # For this we chose a threshold of 0.9 i.e columns with correlation above 90% are considered as highly correlated
    mass_undefined_data, retained_columns_undef = delete_correlated_features(mass_undefined_data, 0.9,
                                                                             retained_columns_undef)
    mass_defined_data, retained_columns_def = delete_correlated_features(mass_defined_data, 0.9, retained_columns_def)

    # For each sub category, we keep track of deleted columns, this will be useful to handle this columns in test phase
    deleted_columns_undef = list(set(COLUMN_NAMES) - set(retained_columns_undef))
    deleted_columns_def = list(set(COLUMN_NAMES) - set(retained_columns_def))

    # For each sub category, we standardize the data and keep track of the means and standard deviations computed
    mass_undefined_data, mean_undef, std_undef = standardize_data(mass_undefined_data)
    mass_defined_data, mean_def, std_def = standardize_data(mass_defined_data)

    # For each sub category, we do a polynomial feature expansion up to the best degree found during cross validation
    mass_undefined_data = data_augmentation(mass_undefined_data, best_degrees[jet_num][1])
    mass_defined_data = data_augmentation(mass_defined_data, best_degrees[jet_num][0])

    # applying the ridge regression algorithm
    w_undef, mse_undef = ridge_regression(mass_undefined_labels, mass_undefined_data, best_lambdas[jet_num][1])
    w_def, mse_def = ridge_regression(mass_defined_labels, mass_defined_data, best_lambdas[jet_num][0])

    # saving the deleted columns, means, standard deviations, and weights
    deleted_columns.append([deleted_columns_def, deleted_columns_undef])
    mean_per_category.append([mean_def, mean_undef])
    std_per_category.append([std_def, std_undef])
    weights_per_category.append([w_def, w_undef])

#######################################################################################################################
# Testing
#######################################################################################################################

# loading the test dataset
_, tX_test, org_ids_test = load_csv_data(DATA_TEST_PATH)

# lists that will hold the event ids and corresponding prediction
res_ids = np.empty(0)
res_y_pred = np.empty(0)

# iterating over each jet
for jet_num in range(CATEGORIES_NBR):
    # For each jet, we isolate the rows of this category
    features, ids_test = data_categorization(tX_test, org_ids_test, jet_num)

    # for each jet, we sub categorize the rows using the first feature i.e the mass_MMC feature
    mass_undefined_data, mass_undefined_ids, mass_defined_data, mass_defined_ids = data_sub_categorization(features,
                                                                                                           ids_test)

    # For each sub category, we delete the columns that was determined during training
    # We also keep track of the columns retained after this first step of cleaning
    mass_undefined_data, retained_columns_undef = delete_columns_from_list(mass_undefined_data, COLUMN_NAMES,
                                                                           deleted_columns[jet_num][1])
    mass_defined_data, retained_columns_def = delete_columns_from_list(mass_defined_data, COLUMN_NAMES,
                                                                       deleted_columns[jet_num][0])

    # For each sub category, we apply the same transformations applied to the training set
    mass_undefined_data = data_transformation(mass_undefined_data, jet_num, retained_columns_undef)
    mass_defined_data = data_transformation(mass_defined_data, jet_num, retained_columns_def)

    # For each sub category, we standardize the data using the means and standard deviations from the train phase
    mass_undefined_data = (mass_undefined_data - mean_per_category[jet_num][1]) / std_per_category[jet_num][1]
    mass_defined_data = (mass_defined_data - mean_per_category[jet_num][0]) / std_per_category[jet_num][0]

    # For each sub category, we do a polynomial feature expansion up to the best degree found during cross validation
    mass_undefined_data = data_augmentation(mass_undefined_data, best_degrees[jet_num][1])
    mass_defined_data = data_augmentation(mass_defined_data, best_degrees[jet_num][0])

    # For each sub category, we generate the predictions
    mass_undefined_predict = predict_labels(weights_per_category[jet_num][1], mass_undefined_data)
    mass_defined_predict = predict_labels(weights_per_category[jet_num][0], mass_defined_data)

    # we save the ids and predictions
    res_ids = np.append(res_ids, mass_undefined_ids)
    res_y_pred = np.append(res_y_pred, mass_undefined_predict)
    res_ids = np.append(res_ids, mass_defined_ids)
    res_y_pred = np.append(res_y_pred, mass_defined_predict)

# creating submission file
OUTPUT_PATH = 'submission.csv'
create_csv_submission(res_ids, res_y_pred, OUTPUT_PATH)
