#!/usr/bin/env python
# coding: utf-8

# In[58]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier
from sklearn.metrics import roc_auc_score
from sklearn import linear_model


# ## Loading the blood donations data

# In[59]:


transfusion = pd.read_csv('transfusion.data',delimiter=',')
transfusion.head()


# ## Inspecting transfusion DataFrame

# In[60]:


transfusion.info()


# ## Creating target column

# In[61]:


transfusion.rename(
    columns={'whether he/she donated blood in March 2007': 'target'},
    inplace=True
)


# ## Checking target incidence

# In[70]:


transfusion.target.value_counts(normalize=True).round(3)


# ## Split transfusion DataFrame into train and test datasets

# In[63]:


X_train, X_test, Y_train, Y_test = train_test_split(
    transfusion.drop(columns='target'),
    transfusion.target,
    test_size=0.25,
    random_state=5,
    stratify=transfusion.target
)


# In[64]:


print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# ## Selecting model using TPOT

# In[65]:


tpot = TPOTClassifier(
    generations=5,
    population_size=20,
    verbosity=2,
    scoring='roc_auc',
    random_state=5,
    disable_update_check=True,
    config_dict='TPOT light'
)
tpot.fit(X_train, y_train)

# AUC score for tpot model
tpot_auc_score = roc_auc_score(y_test, tpot.predict_proba(X_test)[:, 1])
print(f'\nAUC score: {tpot_auc_score:.4f}')

# Printing best pipeline steps
print('\nBest pipeline steps:', end='\n')
for idx, (name, transform) in enumerate(tpot.fitted_pipeline_.steps, start=1):
    print(f'{idx}. {transform}')


# In[66]:


tpot.fitted_pipeline_


# ## Checking the variance

# In[67]:


X_train.var().round(3)


# ## Log normalization

# In[68]:


X_train_normed, X_test_normed = X_train.copy(), X_test.copy()

col_to_normalize = 'Monetary (c.c. blood)' # column to normalize

for df_ in [X_train_normed, X_test_normed]:
    df_['monetary_log'] = np.log(df_[col_to_normalize])
    df_.drop(columns=col_to_normalize, inplace=True)


# In[69]:


X_train_normed.var().round(3) # Checking the variance for X_train_normed


# ## Training regression model

# In[53]:


logreg = linear_model.LogisticRegression(
    solver='liblinear',
    random_state=5
)

logreg.fit(X_train_norm, Y_train) # training the model

# AUC score for tpot model
logreg_auc_score = roc_auc_score(Y_test, logreg.predict_proba(X_test_norm)[:, 1])
print(f'\nAUC score: {logreg_auc_score:.4f}')

