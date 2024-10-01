import joblib
import pandas as pd
import numpy as np

DescionTree_m = joblib.load('cardiovascular_dtree.pkl')

def age_set(column):
    mapped=[]
    for row in column:
        if row>100:
            row = row/365
            mapped.append(row)
        else:
            mapped.append(row)
    return mapped


def predict(data_frame):
    #data_frame= data_frame.drop(columns='id')
    data_frame['age'] = age_set(data_frame['age'])
    data_frame['age'] =  data_frame['age'].astype(int)
    #data_frame= data_frame.drop(columns='cardio')
    
    result1 = DescionTree_m.predict(data_frame)
    val=[]
    a='Patient doesnt have Cardiovascular disease'
    b='Patient have Cardiovascular disease'

    for i in result1:
        if i==0:
            val.append(a)
        else:
            val.append(b)
    return val
        

