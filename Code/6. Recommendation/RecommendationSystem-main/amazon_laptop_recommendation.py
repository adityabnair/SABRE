#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')
print('\nLaptops')
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


product_ratings=pd.read_csv('data/clean_review_laptops.csv', sep=',')


# In[3]:


product_ratings.head()


# In[4]:


product_ratings.shape


# In[5]:


col_list = ['reviewerID', 'asin', 'Rating','title','price']

product_ratings = product_ratings[col_list]

product_ratings.head()


# In[7]:


product_ratings.shape


# In[8]:


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


df1_lap=df[df['userID'].isin(counts1[counts1 >=3].index)]
df1_lap.shape


# In[24]:


df1_lap=df1_lap[df1_lap['prod_ID'].isin(counts[counts >=50].index)]
df1_lap.shape


# In[25]:


df1_lap.head()


# In[27]:


ratings_sum = pd.DataFrame(df1_lap.groupby(['prod_ID'])['rating'].sum()).rename(columns = {'rating': 'ratings_sum'})
top10 = ratings_sum.sort_values('ratings_sum', ascending = False).head(10)
top10


# In[28]:


top10_popular=top10.merge(df1_lap,left_index = True, right_on = 'prod_ID').drop_duplicates(
    ['prod_ID', 'prod_name','price'])[['prod_ID', 'prod_name','ratings_sum','price']]


# In[29]:


print ('Top 10 Popular Products by sum user ratings\n')
top10_popular


# In[30]:


df1_lap.drop_duplicates(subset ="prod_ID", keep = False, inplace = True)


# In[31]:


user_lap_id = df1_lap['userID'].tolist()


# In[32]:


ratingsd=df1_lap.pivot(index='prod_ID',columns= 'userID',values='rating').fillna(0)
ratingsd.head()


# In[33]:


ratingsd.shape


# In[34]:


from sklearn.model_selection import train_test_split
traind, testd = train_test_split(ratingsd, test_size=0.30,random_state=42)


# In[35]:


train = traind.to_numpy()
test = testd.to_numpy()


# In[36]:


sparsity = float(len(train.nonzero()[0]))
sparsity /= (train.shape[0] * train.shape[1])
sparsity *= 100
print ('Sparsity: {:5.2f}%'.format(sparsity))


# In[37]:


def item_similarity(ratings, epsilon=1e-9):
    # epsilon -> for handling dived-by-zero errors
    sim = ratings.T.dot(ratings) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)


# In[38]:


item_sim = item_similarity(train)


# In[39]:


def predict_item(ratings, similarity):
    return ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])


# In[40]:


item_prediction = predict_item(train, item_sim)


# In[41]:


item_prediction[:4, :4]


# In[42]:


from sklearn.metrics import mean_squared_error

def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)


# In[43]:


print ('Item-based CF MSE: ' + str(get_mse(item_prediction, test)))


# In[44]:


from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

svd = TruncatedSVD(n_components=4, n_iter=7, random_state=42)
r_mat_tr=svd.fit_transform(traind) 
print(svd.explained_variance_ratio_)  
print(svd.explained_variance_ratio_.sum())

#pm=pd.DataFrame(cosine_similarity(r_mat_tr))
#pm.head()
ctrain = cosine_similarity(r_mat_tr)


# In[45]:


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


# In[46]:


print (' CF MSE: ' + str(get_mse(ctrain, ctest)))


# In[47]:


df1_lap = df1_lap.sort_values(by='rating')
df1_lap = df1_lap.reset_index(drop=True)
count_users = df1_lap.groupby("userID", as_index=False).count()


# In[48]:


count = df1_lap.groupby("prod_ID", as_index=False).mean()


# In[49]:


items_df_lap = count[['prod_ID']]
items_df_lap.head()
print(len(items_df_lap))


# In[50]:


users_df = count_users[['userID']]
users_df.head()
print(len(users_df))


# In[51]:


users_list = users_df.values
len(users_list)


# In[52]:


df_clean_matrix = df1_lap.pivot(index='prod_ID', columns='userID', values='rating').fillna(0)
df_clean_matrix = df_clean_matrix.T
R = (df_clean_matrix).to_numpy()
R


# In[53]:


user_ratings_mean = np.mean(R, axis = 1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)
R_demeaned


# In[54]:


from scipy.sparse.linalg import svds
U, sigma, Vt = svds(R_demeaned)


# In[55]:


sigma = np.diag(sigma)


# In[56]:


all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df_lap = pd.DataFrame(all_user_predicted_ratings, columns = df_clean_matrix.columns)
preds_df_lap['userID'] = users_df
preds_df_lap.set_index('userID', inplace=True)
preds_df_lap.head(50)


# In[57]:


preds_df_lap.shape


# In[59]:


def recommend_it_lap(predictions_df, itm_df, original_ratings_df, num_recommendations=10,ruserId='A114W0OUE6A4GE'):
    
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
    topk = topk.rename({'prod_ID':'Product ID','prod_name':'Product Name','price':'Price'},axis=1)
    topk.index = np.arange(1, len(topk)+1)
    return topk


# In[60]:


recommend_it_lap(preds_df_lap, items_df_lap, df1_lap, 5,'A23D13HKTA95WX')


# In[61]:


#recommend for any user
recommend_it_lap(preds_df_lap, items_df_lap, df1_lap, 5,'A17R8NRH2UTZ40')


# In[62]:


recommend_it_lap(preds_df_lap, items_df_lap, df1_lap, 5,'A2Q87SAW72XHBB')
