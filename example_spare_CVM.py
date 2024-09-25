#!/gpfs/fs001/cbica/home/tirumals/venv/bin/python
# %%
import pandas as pd
import joblib
import os, sys
import numpy as np

import spare_scores
# %%

mdl_path = './spare_scores/mdl'
df = pd.read_csv('./spare_scores/data/example_data.csv')
df, features = spare_scores.data_prep.prep_spare_cvm(df,mdl_path)

# %%

df_train = df[['ID','DX',]+[ft for ft in features]]
result =  spare_scores.spare.spare_train(  df_train,
                        to_predict= 'DX',
                        model_type= 'SVM_CVM',
                        kernel = 'linear',
                        k=5, 
                        mdl_task = 'Classification',
                        n_repeats=10,
                        pos_group='1',
                        scale_features = False,
                        key_var = 'ID',
                        data_vars=features,
                        output = './spare_train.pkl.gz',
                        verbose=3,
                        log = './log_spare_train.log'
                        )
# %%
from sklearn.utils import shuffle
df_test = shuffle(df_train)
df_test['ID'] = df_test['ID']+100
spare_test = spare_scores.spare.spare_test(df_test,
                            mdl_path    = './spare_train.pkl.gz',
                            key_var     = 'ID',
                            output      =   './spare_test.pkl.gz',
                            verbose = 1)