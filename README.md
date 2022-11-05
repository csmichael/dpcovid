# Code for DASFAA submission 

We publish the source codes for the experiments for the submission. 

## Environment:

Python: 3.9.2 

We install the packages with conda, and the requirements file are exported at `requirements.txt`. 

OS: Linux



## Prepare the dataset:

Since the size of Gowalla dataset is large, please download the dataset and unzip it as follows:

1, Download the Gowalla dataset:
`wget http://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz`

2, Unzip the file:
`tar -zxvf loc-gowalla_totalCheckins.txt.gz`

(Optional) * Synthetic dataset:*
We've prepared the file 'df_synthetic_1600.csv' for basic testing on the synthetic dataset. For the scalability test, since the dataset is too large, readers please generate the datasets with 100K users using:
`python main.py --gen_synthetic`

## To run the experiment:

With default parameters:
`python main.py`

Or change certain control variables:
`python main.py --eps_P=2.0 --eps_u=2.0 --num_patients=2 --num_users=200`
