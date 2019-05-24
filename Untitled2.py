#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score,GridSearchCV,train_test_split
from sklearn.ensemble import GradientBoostingClassifier

list=pd.read_csv('dataset.csv')

X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,y_d,test_size=0.3,random_state=42)

gbc=GradientBoostingClassifier(learning_rate=0.07,n_estimators=123,max_depth=5,min_samples_split=50,min_samples_leaf=5,subsample=0.8,max_features='sqrt',random_state=21)

gbc.fit(X_train,y_train)
y_pred=gbc.predict(X_test)

print(gbc.score(X_test,y_test))



# In[25]:


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score,GridSearchCV,train_test_split
from sklearn.ensemble import GradientBoostingClassifier

list=pd.read_csv('dataset.csv')

X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,y_d,test_size=0.3,random_state=42)

gbc=GradientBoostingClassifier(learning_rate=0.1,n_estimators=125,max_depth=5,min_samples_split=50,min_samples_leaf=5,subsample=0.8,max_features='sqrt',random_state=21)

gbc.fit(X_train,y_train)
y_pred=gbc.predict(X_test)

print(gbc.score(X_test,y_test))



# In[5]:


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

list=pd.read_csv('dataset.csv')

X=list[['Honk_duration','RMS','Intersection density','WiFi density']].values
X_d=pd.DataFrame(X)
y=list['Class'].values
y_d=pd.DataFrame(y)

X_d_2=pd.get_dummies(X_d)
y_d_2=pd.get_dummies(y_d)

X_train, X_test, y_train, y_test = train_test_split(X_d,y_d,test_size=0.3,random_state=42)

gbc=GradientBoostingClassifier(learning_rate=0.05,n_estimators=125,max_depth=5,min_samples_split=50,min_samples_leaf=5,subsample=0.8,max_features='sqrt')

gbc.fit(X_train,y_train)
y_pred=gbc.predict(X_test)

print(gbc.score(X_test,y_test))



# In[6]:


from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

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
    if cond_d.iloc[i,0]>=19.5 and cond_d.iloc[i,0]<21.5:
        y_d.at[i,'SpeedRange']=['1','2']
    if cond_d.iloc[i,0]>=31.5 and cond_d.iloc[i,0]<36.5:
        y_d.at[i,'SpeedRange']=['2','3']
    if cond_d.iloc[i,0]>=48.5 and cond_d.iloc[i,0]<51.5:
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
clf = DecisionTreeClassifier(max_depth=9,min_samples_leaf=6)
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))





# In[3]:


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





# In[ ]:




