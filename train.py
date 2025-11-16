import pickle,os
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

def load_data(path='data/heart.csv'):
    df=pd.read_csv(path)
    df.columns=df.columns.str.lower().str.replace(' ', '_')

    for col in ['restingbp','cholesterol']:
        df[col]=df[col].replace(0, np.nan)
        df[col]=df[col].fillna(df[col].median())
    
    return df

def train_model(df):
    y=df['heartdisease']
    X=df.drop(columns=['heartdisease'])

    df_train,df_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

    dv=DictVectorizer(sparse=False)
    train_dicts=df_train.to_dict(orient='records')
    X_train=dv.fit_transform(train_dicts)
    test_dicts=df_test.to_dict(orient='records')
    X_test=dv.transform(test_dicts)

    xgb=XGBClassifier(
        learning_rate= 0.1,
        max_depth=3,
        n_estimators=100,
        subsample=1.0,
        min_child_weight=1,
        objective='binary:logistic',
        eval_metric='auc',
        nthread=8,
        random_state=1,
        use_label_encoder=False
    )

    xgb.fit(X_train,y_train)

    print('train_auc:',xgb.score(X_train,y_train))
    print('test_auc:',xgb.score(X_test,y_test))

    pipeline=(dv,xgb)
    return pipeline

def save_model(pipeline,path='models/heart_model.bin'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,'wb') as f_out:
        pickle.dump(pipeline,f_out)
    print(f'Model saved to {path}')

if __name__=='__main__':
    df=load_data()
    pipeline=train_model(df)
    save_model(pipeline)