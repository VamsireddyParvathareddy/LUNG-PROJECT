# LUNG-PROJECT,
```
#python,#Machine learning
 import numpy as np import 
pandas as pd import seaborn as sns import matplotlib.pyplot as plt 
from sklearn.metrics 	import 	accuracy_score, 
f1_score,precision_score, recall_score, confusion_matrix 
,ConfusionMatrixDisplay import warnings warnings.filterwarnings("ignore") df=pd.read_csv('data.csv') df df['LUNG_CANCER'].value_counts() df.columns df.info() 
 
df['GENDER']=df['GENDER'].apply(lambda x:1 if x=='M' else 0) 
df.isna().sum() df 
plt.figure(figsize=(15,15)) 
sns.heatmap(df.corr(), annot=True) plt.show() sns.lineplot(data=df.head(50)) 
X = df.drop(['LUNG_CANCER'], axis=1) Y 
= df['LUNG_CANCER'] 
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state= 
70) 
from sklearn.ensemble import RandomForestClassifier 
model = RandomForestClassifier(n_estimators = 100, random_state = 42) model.fit(X_train, y_train) y_pred = model.predict(X_test) print('The accuracy score of this Random Forest Classifier model is: 
{0:.1f}%'.format(100*accuracy_score(y_test, y_pred))) sns.set(rc = {'axes.grid':False}) 
ConfusionMatrixDisplay.from_predictions(y_test, y_pred) plt.title('Confusion matrix of the model') plt.show() from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 2) # n_neighbors means k knn.fit(X_train, y_train) y_pred = knn.predict(X_test) print('The accuracy score of this KNN Classifier model is: 
{0:.1f}%'.format(100*accuracy_score(y_test, y_pred))) sns.set(rc = {'axes.grid':False}) 
ConfusionMatrixDisplay.from_predictions(y_test, y_pred) plt.title('Confusion matrix of the model') plt.show() from sklearn.naive_bayes import GaussianNB nb 
= GaussianNB() nb.fit(X_train, y_train y_pred = nb.predict(X_test) print('The accuracy score of 
this GaussianNB Classifier 	model is: 
{0:.1f}%'.format(100*accuracy_score(y_test, y_pred))) sns.set(rc = {'axes.grid':False}) 
ConfusionMatrixDisplay.from_predictions(y_test, y_pred) plt.title('Confusion matrix of the model') plt.show() 
Result= model.predict(np.array([[1,69,1,2,2,1,1,2,1,2,2,2,2,2,2]])) 
Result[0] 
Result= model.predict(np.array([[1,70,2,1,1,1,1,2,2,2,2,2,2,1,2]])) Result[0] 
```

