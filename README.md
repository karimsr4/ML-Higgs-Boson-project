# CS-433 Machine Learning Project
##### By Ahmed J., Karim H. and Ayman M.

This code is our solution to EPFL's CS-433 Machine Learning course competition :
[EPFL Machine Learning Higgs](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs "EPFL Machine Learning Higgs")

## Results
Our implementation achieves an accuracy of [**81.5%**](https://www.aicrowd.com/f076a64d56fb "Submission #92917").

## Files
- `proj1_helpers.py`: file provided by the teaching team.
- `implementations.py`: contains our implementations of the labs' functions
- `cross_validation.py`: contains our cross validation code
- `cross_validation_run.py`: performs cross validation to produce the best hyper-parameters for our model (stored in
`best_degrees.npy`, `best_lambdas.npy`)
- `data_manipulation.py`: contains our helper methods to perform data manipulation and feature engineering
- `run.py`: contains the code we use to train our model and make our predictions (needs `best_degrees.npy`,
`best_lambdas.npy`)
- `best_degrees.npy`: contains the best degrees for feature augmentation. Can be reproduced by running the script
`cross_validation_run.py`
- `best_lambdas.npy`: contains the best lambda for our machine learning model training. Can be reproduced by running
the script `cross_validation_run.py`
- `submission.csv`: tabular file containing our predictions. Used for the
[submission](https://www.aicrowd.com/f076a64d56fb "Submission #92917")
on the 
[challenge website](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs "EPFL Machine Learning Higgs")

## Execution
To execute our code, the dataset must be downloaded from [here](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/dataset_files "Dataset"). The `.csv` must be extracted from the archive and placed in a folder called `data` placed on the root of the project.

## Contact
In case any help is needed:
- Ahmed Jellouli : ahmed.jellouli@epfl.ch
- Ayman Mezghani : ayman.mezghani@epfl.ch
- Karim Hadidane : karim.hadidane@epfl.ch
