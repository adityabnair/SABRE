#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')
print('\nHeadphones')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
import gzip
#import Recommender as Recommender
#import Evaluation as Evaluation


# In[2]:
pd.set_option('colheader_justify', 'center')
product_ratings=pd.read_csv('data/Reduced_Cleaned_Reviews_headphones.csv', sep=',')


# In[3]:


product_ratings.head()


# In[4]:


product_ratings.shape


# In[5]:


col_list = ['reviewerID', 'asin', 'Rating','title']

product_ratings = product_ratings[col_list]


# In[ ]:





# In[6]:


product_ratings.head()


# In[7]:


product_ratings.shape


product_ratings = product_ratings.rename(columns={'reviewerID': 'userID', 'asin': 'prod_ID','title': 'prod_name',
                                                 'Rating': 'rating' })


# In[9]:


product_ratings.head()


# In[ ]:





# In[10]:


df=product_ratings


# In[11]:


df.isnull().any().any()


# In[12]:


df.isnull().sum().sum()


# In[13]:


df=df.dropna()


# In[14]:


df.shape


# In[15]:


df.isnull().any().any()


# In[16]:


len(df['userID'].unique())


# In[17]:


df.info()


# In[18]:


counts1=df['userID'].value_counts() 
counts=df['prod_ID'].value_counts()


# In[19]:


counts1


# In[20]:


counts


# In[21]:


len(counts1[counts1>=3].index)


# In[22]:


len(counts[counts>=50].index)


# In[23]:


df1_head=df[df['userID'].isin(counts1[counts1 >=3].index)]
df1_head.shape


# In[24]:


df1_head=df1_head[df1_head['prod_ID'].isin(counts[counts >=50].index)]
df1_head.shape


# In[25]:


df1_head.head()


# In[26]:


ratings_sum = pd.DataFrame(df1_head.groupby(['prod_ID'])['rating'].sum()).rename(columns = {'rating': 'ratings_sum'})
top10 = ratings_sum.sort_values('ratings_sum', ascending = False).head(10)
top10


# In[27]:


top10_popular=top10.merge(df1_head,left_index = True, right_on = 'prod_ID').drop_duplicates(
    ['prod_ID', 'prod_name'])[['prod_ID', 'prod_name','ratings_sum']]


# In[28]:


print ('Top 10 Popular Products by sum user ratings\n')
top10_popular


# In[29]:


ratingsd=df1_head.pivot(index='prod_ID',columns= 'userID',values='rating').fillna(0)
ratingsd.head()


# In[30]:


ratingsd.shape


# In[31]:


from sklearn.model_selection import train_test_split
traind, testd = train_test_split(ratingsd, test_size=0.30,random_state=42)


# In[32]:


train = traind.to_numpy()
test = testd.to_numpy()


# In[33]:


sparsity = float(len(train.nonzero()[0]))
sparsity /= (train.shape[0] * train.shape[1])
sparsity *= 100
print ('Sparsity: {:5.2f}%'.format(sparsity))


# In[34]:


def item_similarity(ratings, epsilon=1e-9):
    sim = ratings.T.dot(ratings) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)


# In[35]:


item_sim = item_similarity(train)


# In[36]:


def predict_item(ratings, similarity):
    return ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])


# In[37]:


item_prediction = predict_item(train, item_sim)


# In[38]:


item_prediction[:4, :4]


# In[39]:


from sklearn.metrics import mean_squared_error

def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)


# In[40]:


print ('Item-based CF MSE: ' + str(get_mse(item_prediction, test)))


# In[41]:


from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

svd = TruncatedSVD(n_components=4, n_iter=7, random_state=42)
r_mat_tr=svd.fit_transform(traind) 
print(svd.explained_variance_ratio_)  
print(svd.explained_variance_ratio_.sum())

#pm=pd.DataFrame(cosine_similarity(r_mat_tr))
#pm.head()
ctrain = cosine_similarity(r_mat_tr)


# In[42]:


from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

svd = TruncatedSVD(n_components=4, n_iter=7, random_state=42)
r_mat_tr=svd.fit_transform(testd) 
print(svd.explained_variance_ratio_)  
print(svd.explained_variance_ratio_.sum())

#pmtt=pd.DataFrame(cosine_similarity(r_mat_tr))
#print (pmtt[:2])
#pmtt.head()
ctest = cosine_similarity(r_mat_tr)


# In[43]:


print (' CF MSE: ' + str(get_mse(ctrain, ctest)))


# In[44]:


df1_head = df1_head.sort_values(by='rating')
df1_head = df1_head.reset_index(drop=True)
count_users = df1_head.groupby("userID", as_index=False).count()


# In[45]:


count = df1_head.groupby("prod_ID", as_index=False).mean()


# In[46]:


items_df_head = count[['prod_ID']]
items_df_head.head()
print(len(items_df_head))


# In[47]:


users_df = count_users[['userID']]
users_df.head()
print(len(users_df))


# In[48]:


users_list = users_df.values
len(users_list)


# In[49]:


df_clean_matrix = df1_head.pivot(index='prod_ID', columns='userID', values='rating').fillna(0)
df_clean_matrix = df_clean_matrix.T
R = (df_clean_matrix).to_numpy()
R


# In[50]:


user_ratings_mean = np.mean(R, axis = 1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)
R_demeaned


# In[51]:


from scipy.sparse.linalg import svds
U, sigma, Vt = svds(R_demeaned)


# In[52]:


sigma = np.diag(sigma)


# In[53]:


all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df_head = pd.DataFrame(all_user_predicted_ratings, columns = df_clean_matrix.columns)
preds_df_head['userID'] = users_df
preds_df_head.set_index('userID', inplace=True)
preds_df_head.head(50)


# In[54]:


preds_df_head.shape


# In[55]:


user_head_id = df1_head['userID'].tolist()


# In[56]:


def recommend_it_head(predictions_df, itm_df, original_ratings_df, num_recommendations=10,ruserId='A114W0OUE6A4GE'):
    
    # Get and sort the user's predictions
    sorted_user_predictions = predictions_df.loc[ruserId].sort_values(ascending=False)
    
    # Get the user's data and merge in the item information.
    user_data = original_ratings_df[original_ratings_df.userID == ruserId]
    user_full = (user_data.merge(itm_df, how = 'left', left_on = 'prod_ID', right_on = 'prod_ID').
                     sort_values(['rating'], ascending=False)
                 )

    print ('User {0} has already purchased {1} items.'.format(ruserId, user_full.shape[0]))
    print ('Recommending the highest {0} predicted  items not already purchased.'.format(num_recommendations))
    
    # Recommend the highest predicted rating items that the user hasn't bought yet.
    recommendations = (itm_df[~itm_df['prod_ID'].isin(user_full['prod_ID'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'prod_ID',
               right_on = 'prod_ID').
         rename(columns = {ruserId: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:num_recommendations, :-1]
                      )
    topk=recommendations.merge(original_ratings_df, right_on = 'prod_ID',left_on='prod_ID').drop_duplicates(
    ['prod_ID', 'prod_name'])[['prod_ID', 'prod_name']]
    topk = topk.rename({'prod_ID':'Product ID','prod_name':'Product Name'},axis=1)
    topk.index = np.arange(1, len(topk)+1)
    return topk


# In[57]:


recommend_it_head(preds_df_head, items_df_head, df1_head, 5)


# In[58]:


#recommend for any user
recommend_it_head(preds_df_head, items_df_head, df1_head, 5,'A15U5NUS1EY7IQ')


# In[59]:


recommend_it_head(preds_df_head, items_df_head, df1_head, 5,'A15U5NUS1EY7IQ')


# In[60]:


recommend_it_head(preds_df_head, items_df_head, df1_head, 5,'AZFJMFNXIM3LA')