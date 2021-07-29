# dpcovid_code

We publish the source codes for the experiments.

main.py contains the main codes. 

## Prepare the dataset:

Since the size of Gowalla dataset is large, please download the dataset and unzip it as follows:

1, Download the unzip Gowalla dataset:
`wget http://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz`

2, Unzip the file:
`tar -zxvf loc-gowalla_totalCheckins.txt.gz`

(Optional) * Synthetic dataset:*
We've prepared the file 'df_synthetic_1600.csv' for basic testing on the synthetic dataset. For the scalability test, since the dataset is too large, readers could generate the datasets with 100K users using:
`python main.py --gen_synthetic`

## To run an experiment:

With default parameters:
`python main.py`

Or change certain control variables:
`python main.py --eps_P=2.0 --eps_u=2.0 --num_patients=2 --num_users=200`
