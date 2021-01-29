#importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
%matplotlib inline

#Loading the dataset

digits = load_digits()

#print the data/image dimension: containing 1797 8*8 images 

print('image shape',digits.data.shape)
print('label shape',digits.target.shape)

#ploting some of the images

plt.figure(figsize=(20,6))

for index, (image,label) in enumerate(zip(digits.data[0:5],digits.target[0:5])):
    
    plt.subplot(1,5,index+1)
    plt.imshow(np.reshape(image,(8,8)),cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label)
    
#split the dataset to train and test

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.23, random_state=2)


#print the train/test data/image dimension

print('train set shpe',X_train.shape)
print('test set shape',X_test.shape)


#define a logistic regression model
logistic = LogisticRegression(multi_class='auto')
#fit the model
logistic.fit(X_train,y_train)

#prediction on the test data
predictions = logistic.predict(X_test)

#evaluating the model

#recall, precison and F-scor
print(classification_report(predictions,y_test))

#accuracy
print('The model accuracy is:',logistic.score(X_test,y_test))

#confusion matrix plot
cm = confusion_matrix(predictions,y_test)
plt.figure(figsize=(9,9))
sns.heatmap(cm,annot=True,square=True,cmap='Blues_r',linewidths=0.5)
