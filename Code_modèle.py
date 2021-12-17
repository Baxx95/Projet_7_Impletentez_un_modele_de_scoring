
import pandas as pd
import numpy as np
import time, gc, pickle
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
from sklearn import preprocessing, model_selection, pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from joblib import dump, load


#---------------------------------------------------------------
# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows = None, nan_as_category = False):
    # Read data and merge
    df = pd.read_csv('application_train.csv', nrows= num_rows)
    test_df = pd.read_csv('application_test.csv', nrows= num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    
    del test_df
    gc.collect()
    return df
#---------------------------------------------------------------

appli = application_train_test()

appli = appli[['SK_ID_CURR', 'TARGET', 'CODE_GENDER', 'CNT_CHILDREN', 'FLAG_OWN_CAR',
        'FLAG_OWN_REALTY', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'DAYS_BIRTH', 'AMT_ANNUITY',]]

appli = appli.dropna()

#---------------------------------------------------------------

# definition de stratégie undersample
undersample = RandomUnderSampler(sampling_strategy=0.25)

# Adaptation (entraiment) et application de la transformation
X_under, y_under = undersample.fit_resample(appli.drop(columns='TARGET'), appli.TARGET)

Counter(y_under)

#---------------------------------------------------------------

X_train, X_test, y_train, y_test = model_selection.train_test_split(X_under.drop(columns='SK_ID_CURR'), y_under, train_size=0.8)

#---------------------------------------------------------------

scaler = preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#---------------------------------------------------------------

Random_frst_cls = RandomForestClassifier(random_state=50)

Random_frst_cls.fit(X_train_scaled, y_train)

score = Random_frst_cls.score(X_test_scaled, y_test)

#---------------------------------------------------------------

pipeline = pipeline.Pipeline([('scaler', preprocessing.StandardScaler()), 
                              ('classifier', xgb.XGBClassifier())])

#---------------------------------------------------------------

pipeline.fit(X_train, y_train)

#---------------------------------------------------------------

preds = pipeline.predict(X_test)
Proba_pred = pipeline.predict_proba(X_test)


bd = pd.concat((X_under, y_under), axis=1)
bd['proba_pred'] = Proba_pred[:,1]

#---------------------------------------------------------------

# evaluer les predictions
accuracy = accuracy_score(y_test, preds)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

cm = confusion_matrix(y_test, preds)

#---------------------------------------------------------------
#enregistrer le modèle
dump(pipeline, 'pipeline_Credit_bancaire.joblib') 

#---------------------------------------------------------------



#---------------------------------------------------------------



