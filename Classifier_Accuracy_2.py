#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

road=pd.read_csv('dataset.csv')

road.head()
X=road[['Segment_length','WiFi count','Honk_duration','RMS']]
y=road['Zone']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train) 
knn.score(X_test,y_test)


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

road=pd.read_csv('dataset.csv')

road.head()
X=road[['Segment_length','WiFi count','Honk_duration','RMS']]
y=road['Zone']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
knn=KNeighborsClassifier(n_neighbors=100)
knn.fit(X_train,y_train) 
knn.score(X_test,y_test)


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

road=pd.read_csv('dataset.csv')

road.head()
X=road[['Segment_length','WiFi count','Honk_duration','RMS']]
y=road['Zone']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train) 
knn.score(X_test,y_test)


# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

road=pd.read_csv('dataset.csv')

road.head()
X=road[['Segment_length','WiFi count','Honk_duration','RMS']]
y=road['Zone']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
knn=KNeighborsClassifier(n_neighbors=1000)
knn.fit(X_train,y_train) 
knn.score(X_test,y_test)


# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

road=pd.read_csv('dataset.csv')

road.head()
road_lookup=dict(road.Zone.unique()))
X=road[['Segment_length','WiFi count','Honk_duration','RMS']]
y=road['Zone']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train) 
knn.score(X_test,y_test)
road_prediction=knn.predict([[181.365898,5,0,6.35698]])
road_lookup[road_prediction[0]]


# In[11]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

road=pd.read_csv('dataset.csv')

road.head()
X=road[['Segment_length','WiFi count','Honk_duration','RMS']]
y=road['Zone']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train) 
knn.score(X_test,y_test)
plt.title('training class')
plt.plot(X_train,y_train)
plt.show()


# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
 
road=pd.read_csv('dataset.csv')

road.head()
'''road_lookup=road.Zone.unique()'''
X=road[['Segment_length','WiFi count','Honk_duration','RMS']]
y=road['Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train) 
knn.score(X_test,y_test)
'''road_prediction=knn.predict([[181.365898,5,0,6.35698]])
road_lookup[road_prediction[0]]'''
plt.title('training class')
plt.scatter(X_train,y_train,marker='o')
plt.show()


# In[24]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
 
road=pd.read_csv('dataset.csv')

road.head()
X=road[['Segment_length','WiFi count','Honk_duration','RMS']]
y=road['Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train) 
knn.score(X_test,y_test)


# In[54]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
road=pd.read_csv('dataset.csv')

road.head()
'''road_lookup=road.Zone.unique()'''
X=road[['Segment_length','WiFi count','Honk_duration','RMS']]
y=road['Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
'''knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train) 
knn.score(X_test,y_test)
road_prediction=knn.predict([[181.365898,5,0,6.35698]])
road_lookup[road_prediction[0]]'''
k_range=np.arange(1,101)
train_accuracy=np.empty(len(k_range))
test_accuracy=np.empty(len(k_range))
for i,k in enumerate(k_range):
	knn=KNeighborsClassifier(n_neighbors=k)
	knn.fit(X_train,y_train)
	train_accuracy[i]=knn.score(X_train,y_train)
	test_accuracy[i]=knn.score(X_test,y_test)
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(k_range, test_accuracy, label = 'Testing Accuracy')
plt.plot(k_range, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(np.arange(0,105,5))
plt.show()


# In[84]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
road=pd.read_csv('dataset.csv')

road.head()
'''road_lookup=road.Zone.unique()'''
X=road[['Segment_length','WiFi count','Intersection count','RMS','Timelevel','Honk_duration']]
y=road['Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
'''knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train) 
knn.score(X_test,y_test)
road_prediction=knn.predict([[181.365898,5,0,6.35698]])
road_lookup[road_prediction[0]]'''
k_range=np.arange(1,100)
train_accuracy=np.empty(len(k_range))
test_accuracy=np.empty(len(k_range))
for i,k in enumerate(k_range):
	knn=KNeighborsClassifier(n_neighbors=k)
	knn.fit(X_train,y_train)
	train_accuracy[i]=knn.score(X_train,y_train)
	test_accuracy[i]=knn.score(X_test,y_test)
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(k_range, test_accuracy, label = 'Testing Accuracy')
plt.plot(k_range, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(np.arange(0,100,5))
plt.show()


# In[53]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
 
road=pd.read_csv('dataset.csv')

road.head()
X=road[['Segment_length','WiFi count','Intersection count','RMS']]
y=road['Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
knn=KNeighborsClassifier(n_neighbors=42)
knn.fit(X_train,y_train) 
knn.score(X_test,y_test)


# In[62]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
 
road=pd.read_csv('dataset.csv')

road.head()
X=road[['Segment_length','WiFi count','Honk_duration','RMS']]
y=road['Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
knn=KNeighborsClassifier(n_neighbors=42)
knn.fit(X_train,y_train) 
knn.score(X_test,y_test)


# In[111]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
road=pd.read_csv('dataset.csv')
X=road[['Segment_length','WiFi count','Honk_duration','RMS']]
y=road['Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
clf=DecisionTreeClassifier(max_depth=3).fit(X_train,y_train)
print('training set accuracy: {:.2f}'.format(clf.score(X_train,y_train)))
print('test set accuracy: {:.2f}'.format(clf.score(X_test,y_test)))


# In[108]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
from adspy_shared_utilities import plot_decision_tree
road=pd.read_csv('dataset.csv')
X=road[['Segment_length','Honk_duration','RMS']]
y=road['Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=4)
clf=DecisionTreeClassifier(max_depth=7).fit(X_train,y_train)
print('training set accuracy: {:.2f}'.format(clf.score(X_train,y_train)))
print('test set accuracy: {:.2f}'.format(clf.score(X_test,y_test)))


# In[493]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
from adspy_shared_utilities import plot_decision_tree
from adspy_shared_utilities import plot_feature_importances
road=pd.read_csv('dataset.csv')
X=road[['Segment_length','WiFi count','Honk_duration','RMS','Intersection count','Timelevel']]
y=road['Zone']
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.25)
clf=DecisionTreeClassifier(max_depth=5,min_samples_split=2).fit(X_train,y_train)
print('training set accuracy: {:.2f}'.format(clf.score(X_train,y_train)))
print('test set accuracy: {:.2f}'.format(clf.score(X_test,y_test)))


# In[507]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
 
road=pd.read_csv('dataset.csv')
X=road[['Route','Segment_length','WiFi count','Intersection count','RMS','Timelevel','Honk_duration']]
y=road['Zone']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=25,test_size=0.25)
knn=KNeighborsClassifier(n_neighbors=9,p=1,weights='distance')
knn.fit(X_train,y_train) 
knn.score(X_test,y_test)


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
from adspy_shared_utilities import plot_decision_tree
from adspy_shared_utilities import plot_feature_importances


road=pd.read_csv('dataset.csv')
X=road[['Route','Timelevel','Segment_length','WiFi count','RMS','Honk_duration']]
y=road['Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=15,test_size=0.25)
clf=DecisionTreeClassifier(max_depth=5,min_samples_split=2,max_leaf_nodes=4000).fit(X_train,y_train)
print('training set accuracy: {:.2f}'.format(clf.score(X_train,y_train)))
print('test set accuracy: {:.2f}'.format(clf.score(X_test,y_test)))
print(clf.feature_importances_)
pyplot.bar(range(len(clf.feature_importances_)), clf.feature_importances_)
pyplot.show()


# In[490]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
 
road=pd.read_csv('dataset.csv')
X=road[['Route','Segment_length','WiFi count','Intersection count','RMS','Timelevel','Honk_duration']]
y=road['Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
knn=KNeighborsClassifier(n_neighbors=32,p=1)
knn.fit(X_train,y_train) 
knn.score(X_test,y_test)


# In[197]:


road.head()


# In[509]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
road=pd.read_csv('dataset.csv')
X=road[['Segment_length','WiFi count','Intersection count','RMS','Timelevel','Honk_duration']]
y=road['Zone']
k_range=np.arange(1,100)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=25)
train_accuracy=np.empty(len(k_range))
test_accuracy=np.empty(len(k_range))
for i,k in enumerate(k_range):
	knn=KNeighborsClassifier(n_neighbors=k,p=1,metric='minkowski')
	knn.fit(X_train,y_train)
	train_accuracy[i]=knn.score(X_train,y_train)
	test_accuracy[i]=knn.score(X_test,y_test)
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(k_range, test_accuracy, label = 'Testing Accuracy')
plt.plot(k_range, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(np.arange(0,100,5))
plt.show()


# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
road=pd.read_csv('dataset.csv')
X=road[['Route','Segment_length','WiFi count','Intersection count','RMS','Timelevel','Honk_duration']]
y=road['Class']
k_range=np.arange(1,100)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
train_accuracy=np.empty(len(k_range))
test_accuracy=np.empty(len(k_range))
for i,k in enumerate(k_range):
	knn=KNeighborsClassifier(n_neighbors=k,p=2,metric='minkowski',weights='uniform')
	knn.fit(X_train,y_train)
	train_accuracy[i]=knn.score(X_train,y_train)
	test_accuracy[i]=knn.score(X_test,y_test)
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(k_range, test_accuracy, label = 'Testing Accuracy')
plt.plot(k_range, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(np.arange(0,100,5))
plt.show()


# In[23]:


import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

list=pd.read_csv('dataset.csv')

X=list[['Timelevel','Honk_duration','RMS','Intersection count','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,y_d,test_size=0.3,random_state=42)

model=XGBClassifier(eta=0.05,n_estimators=121,max_depth=7,min_samples_split=50,min_samples_leaf=5,cv=5)

model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(model.feature_importances_)
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
pyplot.show()
print(model.score(X_test,y_test))


# In[ ]:


# Import libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV,train_test_split
from time import time
import matplotlib.pyplot as plt
from operator import itemgetter

# Stop deprecation warnings from being printed
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#predicting accuracy
list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection count','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
print("c")
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,y_d_2,test_size=0.25,random_state=42)
print("b")
feature=X_train.values
clf=RandomForestClassifier(random_state=84)


# In[77]:


# function takes a RF parameter and a ranger and produces a plot and dataframe of CV scores for parameter values
def evaluate_param(parameter, num_range, index):
    grid_search = GridSearchCV(clf, param_grid = {parameter: num_range})
    grid_search.fit(X_train,y_train)
    
    df = {}
    for i, score in enumerate(grid_search.cv_results_):
        df[score[0][parameter]] = score[1]
       
    
    df = pd.DataFrame.from_dict(df, orient='index')
    df.reset_index(level=0, inplace=True)
    df = df.sort_values(by='index')
 
    plt.subplot(3,2,index)
    plot = plt.plot(df['index'], df[0])
    plt.title(parameter)
    return plot, df


# In[83]:


# parameters and ranges to plot
param_grid = {"n_estimators": np.arange(2, 300, 2),
              "max_depth": np.arange(1, 28, 1),
              "min_samples_split": np.arange(1,150,1),
              "min_samples_leaf": np.arange(1,60,1),
              "max_leaf_nodes": np.arange(2,60,1),
              "min_weight_fraction_leaf": np.arange(0.1,0.4, 0.1)}


# In[87]:


index = 1
plt.figure(figsize=(16,12))
for parameter, param_range in dict.items(param_grid):   
    evaluate_param(parameter, param_range, index)
    index += 1


# In[86]:


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

list=pd.read_csv('dataset.csv')

X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,y_d,test_size=0.4,random_state=42)

gbc=GradientBoostingClassifier(learning_rate=0.1,n_estimators=125,max_depth=6,min_samples_split=3,min_samples_leaf=3)

gbc.fit(X_train,y_train)
y_pred=gbc.predict(X_test)

print(gbc.score(X_test,y_test))



# In[35]:


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

list=pd.read_csv('dataset.csv')

X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
y_d=pd.DataFrame(y)
X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,y_d,test_size=0.3,random_state=21)

gbc=GradientBoostingClassifier(learning_rate=0.05,n_estimators=125,max_depth=5,min_samples_split=50,min_samples_leaf=5,subsample=0.8,max_features='sqrt')

gbc.fit(X_train,y_train)
y_pred=gbc.predict(X_test)

print(gbc.score(X_test,y_test))



# In[95]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

list=pd.read_csv('dataset.csv')

X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,y_d,test_size=0.3,random_state=42)

gbc=GradientBoostingClassifier(learning_rate=0.05,n_estimators=121,max_depth=5,min_samples_split=50,min_samples_leaf=5,subsample=0.8,max_features='sqrt')

gbc.fit(X_train,y_train)
y_pred=gbc.predict(X_test)

print(gbc.score(X_test,y_test))
#y_pred = clf.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[58]:


import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier

list=pd.read_csv('dataset.csv')

X=list[['Honk_duration','RMS','Intersection count','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,y_d,test_size=0.3,random_state=42)

#model=XGBClassifier(eta=0.05,n_estimators=121,max_depth=7,min_samples_split=50,min_samples_leaf=5,cv=5)
model=SGDClassifier(loss="hinge",penalty="none",max_iter=1500,n_jobs=1,learning_rate="optimal",alpha=0.1)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
#print(model.feature_importances_)
#pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
#pyplot.show()
print(model.score(X_test,y_test))


# In[60]:


import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

list=pd.read_csv('dataset.csv')

X=list[['Timelevel','Honk_duration','RMS','Intersection count','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,y_d,test_size=0.3,random_state=42)

model=XGBClassifier(eta=0.05,n_estimators=121,max_depth=7,min_samples_split=50,min_samples_leaf=5,cv=5)

model.fit(X_train,y_train)
y_pred=model.predict(X_test)


# In[61]:


import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

list=pd.read_csv('dataset.csv')

X=list[['Timelevel','Honk_duration','RMS','Intersection count','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,y_d,test_size=0.3,random_state=42)

model=XGBClassifier(eta=0.05,n_estimators=121,max_depth=7,min_samples_split=50,min_samples_leaf=5,cv=5)

model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(model.score(X_test,y_test))


# In[62]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection count','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
print("c")
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,y_d_2,test_size=0.3,random_state=42)
print("b")

k_range=np.arange(100,201)
train_accuracy=np.empty(len(k_range))
test_accuracy=np.empty(len(k_range))
for i,k in enumerate(k_range):
	clf=RandomForestClassifier(n_estimators=k,min_samples_split=8,max_depth=10)
	clf.fit(X_train,y_train)
	train_accuracy[i]=clf.score(X_train,y_train)
	test_accuracy[i]=clf.score(X_test,y_test)
plt.title('RANDOM FOREST CLASSIFIER')
plt.plot(k_range, test_accuracy, label = 'Testing Accuracy')
plt.plot(k_range, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(np.arange(100,200,5))
plt.show()

clf.fit(X_train,y_train)
print("a")
y_pred=clf.predict(X_test)
print(y_pred)
print(clf.score(X_test,y_test))


# In[89]:


import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection count','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
print("c")
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,y_d_2,test_size=0.3,random_state=42)
print("b")
clf=RandomForestClassifier(n_estimators=189,max_depth=7,min_samples_split=8,max_features='sqrt',min_samples_leaf=3)
clf.fit(X_train,y_train)
print("a")
y_pred=clf.predict(X_test)
print(y_pred)
print(clf.score(X_test,y_test))


# In[94]:


import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection count','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
print("c")
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,y_d_2,test_size=0.3,random_state=42)
print("b")
clf=RandomForestClassifier(n_estimators=105,max_depth=9,min_samples_split=7,max_features='auto')
clf.fit(X_train,y_train)
print("a")
print(clf.score(X_test,y_test))


# In[90]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state = 42)
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())


# In[114]:


from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 20)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(4, 15, num = 10)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)


# In[118]:


rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)


# In[119]:


rf_random.best_params_


# In[105]:


import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection count','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
print("c")
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,y_d_2,test_size=0.3,random_state=42)
print("b")
clf=RandomForestClassifier(n_estimators= 600,
 min_samples_split= 5,
 min_samples_leaf=2,
 max_features='sqrt',
 max_depth=10,
 bootstrap=False)
clf.fit(X_train,y_train)
print("a")
print(clf.score(X_test,y_test))


# In[117]:


import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection count','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
print("c")
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,y_d_2,test_size=0.3,random_state=42)
print("b")
clf=RandomForestClassifier(n_estimators= 147,
 min_samples_split= 2,
 min_samples_leaf=2,
 max_features='auto',
 max_depth=8,
 bootstrap=True)
clf.fit(X_train,y_train)
print("a")
print(clf.score(X_test,y_test))


# In[134]:


import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

list=pd.read_csv('dataset.csv')
X=list[['Honk_duration','RMS','Intersection count','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
print("c")
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
X_train, X_test, y_train, y_test = train_test_split(X_d,y_d_2,test_size=0.3,random_state=42)
print("b")
clf=RandomForestClassifier(n_estimators= 668,
 min_samples_split= 10,
 min_samples_leaf=1,
 max_features='auto',
 max_depth=8,
 bootstrap=True,random_state=15)
clf.fit(X_train,y_train)
print("a")
print(clf.score(X_test,y_test))


# In[1]:


from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

list=pd.read_csv('6mar.csv')

cond=list['Mean_speed_kmph'].values
cond_d=pd.DataFrame(cond)
X=list[['Honk_duration','Road_surface','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=list[['SpeedRange']].values
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
#print(y_d_2)
#print(cond_d)
y_d.loc[cond_d[0]<=22.5,'SpeedRange']=['1']
y_d.loc[cond_d[0]>22.5,'SpeedRange']=['2']
y_d.loc[cond_d[0]>37.5,'SpeedRange']=['3']
y_d.loc[cond_d[0]>52.5,'SpeedRange']=['4']
i=0
#range=5328
while i<5328:
    if cond_d.iloc[i,0]>=18.5 and cond_d.iloc[i,0]<22.5:
        y_d.at[i,'SpeedRange']=['1','2']
    if cond_d.iloc[i,0]>=32.5 and cond_d.iloc[i,0]<37.5:
        y_d.at[i,'SpeedRange']=['2','3']
    if cond_d.iloc[i,0]>=47.5 and cond_d.iloc[i,0]<52.5:
        y_d.at[i,'SpeedRange']=['3','4']
    i=i+1
    
    
#print(y_d['SpeedRange'])
new_y=y_d['SpeedRange'].copy()
new_y_d=pd.DataFrame(new_y)
#print(new_y)
y_l=new_y.values.tolist()

#print(y_l)
#print(y_l.type())
new_y_d2=MultiLabelBinarizer().fit_transform(y_l)
#print(new_y_d2)

X_train,X_test,y_train,y_test=train_test_split(X_d,y_d_2,test_size=0.4,random_state=21)
gbc=GradientBoostingClassifier(learning_rate=0.05,n_estimators=125,max_depth=5,min_samples_split=50,min_samples_leaf=5,subsample=0.8,max_features='sqrt')

gbc.fit(X_train,y_train)
print(gbc.score(X_test,y_test))





# In[2]:


from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score

list=pd.read_csv('6mar.csv')

cond=list['Mean_speed_kmph'].values
cond_d=pd.DataFrame(cond)
X=list[['Honk_duration','Road_surface','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=list[['SpeedRange']].values
y_d=pd.DataFrame(y)
y_d_2=pd.get_dummies(y_d)
#print(y_d_2)
#print(cond_d)
y_d.loc[cond_d[0]<=22.5,'SpeedRange']=['1']
y_d.loc[cond_d[0]>22.5,'SpeedRange']=['2']
y_d.loc[cond_d[0]>37.5,'SpeedRange']=['3']
y_d.loc[cond_d[0]>52.5,'SpeedRange']=['4']
i=0
#range=5328
while i<5328:
    if cond_d.iloc[i,0]>=18.5 and cond_d.iloc[i,0]<22.5:
        y_d.at[i,'SpeedRange']=['1','2']
    if cond_d.iloc[i,0]>=32.5 and cond_d.iloc[i,0]<37.5:
        y_d.at[i,'SpeedRange']=['2','3']
    if cond_d.iloc[i,0]>=47.5 and cond_d.iloc[i,0]<52.5:
        y_d.at[i,'SpeedRange']=['3','4']
    i=i+1
    
    
#print(y_d['SpeedRange'])
new_y=y_d['SpeedRange'].copy()
new_y_d=pd.DataFrame(new_y)
#print(new_y)
y_l=new_y.values.tolist()

#print(y_l)
#print(y_l.type())
new_y_d2=MultiLabelBinarizer().fit_transform(y_l)
#print(new_y_d2)

X_train,X_test,y_train,y_test=train_test_split(X_d,new_y_d2,test_size=0.3,random_state=21)
model = XGBClassifier(eta=0.5,max_depth=5,seed=21,sub_sample=0.5,nthread=3)
model.fit(X_train, y_train)
#y_pred = model.predict(X_test)
#predictions = [round(value) for value in y_pred]
#accuracy = accuracy_score(y_test, predictions)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))

print(model.score(X_test,y_test))





# In[ ]:


import pandas as pd
import numpy as np

df=pd.read_csv('6mar.csv')

cond=df['Honk_duration']

