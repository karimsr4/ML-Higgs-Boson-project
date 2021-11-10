# This is the main script to be run to get the best hyper-parameters of the model

# necessary importations
import os
from cross_validation import *
from data_manipulation import *
from proj1_helpers import *

# path to the dataset files
DATA_TRAIN_PATH = "data" + os.path.sep + "train.csv"

# loading the train dataset
train_labels, train_set, ids = load_csv_data(DATA_TRAIN_PATH)

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

best_lambdas = []
best_degrees = []

# iterating over each jet
for jet_num in range(CATEGORIES_NBR):
    # For each jet, we isolate the rows of this category
    features, labels = data_categorization(train_set, train_labels, jet_num)

    # for each jet, we sub categorize the rows using the first feature i.e the mass feature
    mass_undefined_data, mass_undefined_labels, mass_defined_data, mass_defined_labels = data_sub_categorization(
        features, labels)

    retained_columns_def = COLUMN_NAMES.copy()
    retained_columns_undef = COLUMN_NAMES.copy()

    # For each sub category, we remove the columns that have unique value -999 i.e undefined, as these columns carry no
    # information
    # We also keep track of the columns retained after this first step of cleaning
    mass_undefined_data, retained_columns_undef = undefined_value_cleaning(mass_undefined_data, retained_columns_undef)
    mass_defined_data, retained_columns_def = undefined_value_cleaning(mass_defined_data, retained_columns_def)

    # For each sub category, we remove the columns that have unique value jet_num i.e the columns PRI_jet_num,
    # This removes also some columns with unique values as PRI_jet_num. For example, last column with PRI_jet_num=0
    mass_undefined_data, retained_columns_undef = category_column_delete(mass_undefined_data, jet_num,
                                                                         retained_columns_undef)
    mass_defined_data, retained_columns_def = category_column_delete(mass_defined_data, jet_num, retained_columns_def)

    # For each sub category, we apply some transformations on the features, mainly applying logarithm function on right
    # skewed distributions, This choice was done after visualizing distributions for each category and selecting the
    # columns to apply log
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

    # For each sub_category, we find the best hyper-parameters for ridge regression and polynomial feature expansion
    best_degree_def, best_lambda_def, loss_def, accuracy_def, f1_score_def = find_best_param_ridge_reg(
        mass_defined_labels, mass_defined_data, 7, -10, -7, 11)

    best_degree_undef, best_lambda_undef, loss_undef, accuracy_undef, f1_score_undef = find_best_param_ridge_reg(
        mass_undefined_labels, mass_undefined_data, 7, -10, -7, 11)

    print(
        f"{best_degree_def} accuracy of category {jet_num} defined= {accuracy_def}, {best_degree_undef} accuracy of category {jet_num} undefined= {accuracy_undef}, mse_def {loss_def} mse_undef {loss_undef} ")

    # saving the best hyper-parameters
    best_degrees.append([best_degree_def, best_degree_undef])
    best_lambdas.append([best_lambda_def, best_lambda_undef])

# save best hyper-parameters to npy files, that will be read in model training
np.save("best_degrees.npy", best_degrees)
np.save("best_lambdas.npy", best_lambdas)
