# -*- coding: utf-8 -*-
"""Naive Bayes.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IJwmIxc-xniDqT7SockCGFQ6b8Ax4Z7q
"""

from google.colab import drive
drive.mount('/content/gdrive')

import pandas as pd

#dataframe
data = pd.read_csv('/content/gdrive/MyDrive/Dataset/kr-vs-kp.csv')

print(data.columns)

from sklearn.model_selection import train_test_split

data_x =  data.loc[:, data.columns != 'label']
data_y = data['label']


train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, train_size=0.75, stratify=data_y)

print(train_x.shape)
print(test_x.shape)

print(train_y.shape)
print(test_y.shape)

print(train_x)
print(train_y)

print(data_x)
print(data_y)

pre1 = 0.6666666666666666
rec1 = 0.5925925925925926
f11 = 0.627450980392157

#Implementation the fit function one
count1 = 0
count2 = 0
for i in data_y:
  if i == 'won':
    count1 = count1 + 1
  else:
    count2 = count2 + 1

won = count1/len(data_y)
nowin = count2/len(data_y)
print(count1, count2)
print(won, nowin)
x = won + nowin
print(x)

#Implementation the fit function two

def fit_function(data, arr):
  #arr = []
  flag = 0
  temp = 0
  for d in data:
    if d != 'label':
      arr.append({})
      X = data.groupby([d, 'label'], as_index=False)['label'].count()
      mylist = list(dict.fromkeys(X[d]))
      for j in mylist:
        arr[flag][j] = []
        for i in range(2):
          try:
            #print(f"look=={temp}, {X['label'][temp]}")
            if temp <= 1:
              arr[flag][j].append(X['label'][temp]/count1)
            else:
              arr[flag][j].append(X['label'][temp]/count2)
            temp = temp + 1
          except:
            pass
      temp = 0
      flag = flag + 1
    


  for ar in arr:
    print(ar)

#Implement the predict function
def predict_function(data, classify):
  arr1 = []
  arr2 = []
  temp1 = 1
  temp2 = 1
  clock = 0
  for i in range(len(data)):
    temp = data.iloc[i]
    for j in temp[0:-1]:
      try:
        temp1 = temp1 * arr[clock][j][0]
        temp2 = temp2 * arr[clock][j][1]
      except:
        temp1 = temp1 * arr[clock][j][0]
        temp2 = temp2 * 1
      clock = clock + 1
    arr1.append(temp1*won)
    arr2.append(temp2*nowin)
    temp1 = 1
    temp2 = 1
    clock = 0
    #print(arr1)
    #print(arr2)

    #print(len(arr2))
    #print(len(arr1))

  for i in range(len(arr1)):
    if arr1[i] > arr2[i]:
      classify.append('won')
    else:
      classify.append('nowin')

arr = []
fit_function(data, arr)

classify = []
print(len(classify))
predict_function(test_x, classify)

print(len(classify))
print(classify)
test_y = test_y.tolist()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

acc1 = accuracy_score(test_y, classify)
rec1 = recall_score(test_y, classify)
pre1 = precision_score(test_y, classify)
f11 = f1_score(test_y, classify)

print(acc1, pre1, rec1, f11)