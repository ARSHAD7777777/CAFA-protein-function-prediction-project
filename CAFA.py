#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import necessary packages
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.compose import make_column_transformer
import matplotlib.pyplot as plt
import seaborn as sns


# In[18]:


print('tensorflow version',tf.__version__)


# # Loading and pre-processing the data

# In[23]:


train_data=pd.read_csv("/kaggle/input/cafa-5-protein-function-prediction/Train/train_terms.tsv",sep="\t")


# In[24]:


train_data.columns


# In[25]:


train_data


# In[9]:


embeds=np.load('/kaggle/input/t5embeds/train_embeds.npy')


# In[10]:


embeds.shape


# In[3]:


protein_ids=np.load('/kaggle/input/t5embeds/train_ids.npy')


# In[ ]:


embeddings=pd.DataFrame(embeds,columns=['Number'+str(i) for i in range(1,1025)])


# In[ ]:


embeddings.index=protein_ids


# In[ ]:


embeddings


# In[ ]:


train_new=pd.DataFrame(index=['Number'+str(i) for i in range(1,1025)])


# In[ ]:


for i in train_data.EntryID:
    train_new[str(i)]=embeddings.loc[str(i)]


# In[ ]:


train_new=train_new.transpose()


# In[70]:


train_new.head(5)


# In[26]:


y=pd.DataFrame(train_data.term)


# In[27]:


y_freq=pd.DataFrame(y.value_counts(),columns=['frequency'])


# In[28]:


y_freq


# In[29]:


l=y_freq[y_freq.frequency>=500].index.tolist()


# In[50]:


y_freq[y_freq.frequency>=500].index.get_level_values(0)


# In[60]:


freq_term=[]
for i in range(len(l)):
    freq_term.append(l[i][0])


# In[61]:


len(freq_term)


# In[62]:


df=train_data["term"].isin(freq_term)


# In[63]:


train_data_new=train_data[df]


# In[64]:


train_data_new


# In[65]:


sns.countplot(train_data_new.aspect)


# In[71]:


a=train_new.index.isin(list(train_data_new.EntryID.values))


# In[72]:


train_new=train_new[a]


# In[ ]:


le=LabelEncoder()


# In[ ]:


df_y=train_data_new[['EntryID','term']]
df.set_index('EntryID',inplace=True)
y=le.fit_transform(list(df_y.term))


# In[ ]:


transformer = make_column_transformer(
    (OneHotEncoder(), ['term']),
    remainder='passthrough')


# In[ ]:


transformed = transformer.fit_transform(df_y)


# In[ ]:


transformed


# In[ ]:


df_y_new=pd.DataFrame.sparse.from_spmatrix(transformed,columns=transformer.get_feature_names(),index=df_y.index)


# In[ ]:


y_list=[]
for i in list(train_new.index):
    array=np.sum(np.array(df_new.loc[i],),axis=0)
    y_list.append(array)


# In[ ]:


a=-1
for i in arr_list:
    a=a+1
    if len(i.shape)==0:
        y_list[a]=np.array(df_y_new.loc[list(train_new.index)[a]])


# In[8]:


from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner


# In[28]:


train_new


# # Splittting the data into training and testing sets

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(train_new, y_list, test_size=0.20, random_state=42)


# In[ ]:


x_train=np.array(x_train)
x_test=np.array(x_test)
y_train=np.array(y_train)
y_test=np.array(y_test)


# In[ ]:


x_train=pd.DataFrame(x_train,columns=train_new.columns)
x_test=pd.DataFrame(x_test,columns=train_new.columns)
y_train=pd.DataFrame(y_train,columns=df_new.columns)
y_test=pd.DataFrame(y_test,columns=df_new.columns)


# In[ ]:


x_train=pd.DataFrame(x_train,columns=train_new.columns)
x_test=pd.DataFrame(x_test,columns=train_new.columns)
y_train=pd.DataFrame(y_train,columns=df_new.columns)
y_test=pd.DataFrame(y_test,columns=df_new.columns)


# # Building an ANN model

# In[8]:


def build_model(hp):
    model = keras.Sequential()
    model.add(layers.BatchNormalization(input_shape=[1024]))
    for i in range(hp.Int("num_layers", 1, 4)):
        model.add(layers.Dense(units=hp.Int(f"units_{i}", min_value=500, max_value=1000, step=120),
                               activation=hp.Choice("activation", ["relu", "tanh"])))
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=0.25))
        
    model.add(layers.Dense(units=1324,
                           activation='softmax'))
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=['accuracy',"binary_accuracy",keras.metrics.AUC()],
    )
    return model

build_model(keras_tuner.HyperParameters())


# In[10]:


model = keras.Sequential()
model.add(layers.BatchNormalization(input_shape=[1024]))
model.add(layers.Dense(units=740,activation='relu'))
model.add(layers.Dense(units=860,activation='relu'))
model.add(layers.Dense(units=1324,activation='softmax'))
model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=['accuracy',"binary_accuracy",keras.metrics.AUC()],
    )


# # Hyperparameter tuning

# In[9]:


tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=5,
    executions_per_trial=1,
    overwrite=True,
    directory="C:\\Users\\arshad\\Downloads\\train_ids.npy",
    project_name="helloworld",
)


# In[10]:


tuner.search_space_summary()


# In[11]:


tuner.search(x_train, y_train, 
             epochs=3, 
             validation_data=(x_test,y_test))


# # Assesing the best model

# In[15]:


models = tuner.get_best_models(num_models=2)
best_model = models[0]


# In[16]:


best_model.summary()


# In[93]:


test_data[0].shape


# In[94]:


best_model.build(input_shape=(1024,))


# In[ ]:


best_model.fit(x_train,y_train,epochs=5)


# In[90]:


test_protein_ids = np.load('/kaggle/input/t5embeds/test_ids.npy')
test_data=np.load('/kaggle/input/t5embeds/test_embeds.npy')


# # Predicting with the best model

# In[ ]:


predictions=best_model.predict(test_data)


# In[ ]:


predictions


# In[ ]:


predictions.set_index(test_protein_ids,inplace=True)


# In[ ]:


predictions.columns=freq_term


# In[ ]:


predictions


# In[ ]:


df_submission=predictions.stack()


# In[ ]:


df_submission = pd.DataFrame({'Protein Id': df_submission.index.get_level_values(0),
                              'GO Term Id': df_submission.index.get_level_values(1),
                              'Prediction': df_submission.values
                             })


# In[9]:


df_submission


# In[10]:


df_submission.to_csv('df_submission.tsv',header=False, index=False, sep="\t")

