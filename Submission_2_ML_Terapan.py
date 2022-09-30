#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
import warnings

plt.style.use("fivethirtyeight")
warnings.filterwarnings("ignore")


# In[4]:


rating = pd.read_csv('C:/Users/ASUS/Downloads/ratings.csv')
movies = pd.read_csv('C:/Users/ASUS/Downloads/movies_metadata.csv')
keywords = pd.read_csv('C:/Users/ASUS/Downloads/keywords.csv')
credits = pd.read_csv('C:/Users/ASUS/Downloads/credits.csv')

print('Rating shape:', rating.shape)
print('Movies shape:', movies.shape)
print('Keywords shape:', keywords.shape)
print('Credits shape:', credits.shape)


# In[5]:


rating.head()


# In[6]:


movies.head()


# In[7]:


keywords.head()


# In[8]:


credits.head()


# In[9]:


movies.info()


# In[10]:


movies.describe()


# In[11]:


movies_md = movies[movies['vote_count'] >= 110]


# In[12]:


movies_md.head()


# In[13]:


movies_md.columns


# In[14]:


movies_md = movies_md[['id', 'original_title', 'overview', 'genres', 'release_date', 'runtime']]


# In[15]:


movies_md.head()


# In[16]:


movies_md.reset_index(inplace=True, drop=True)


# In[17]:


movies_md.head()


# In[18]:


movies_credits = credits[['id', 'cast']]


# In[19]:


movies_credits.head()


# In[20]:


movies_rating = rating[['movieId', 'rating']]
movies_rating.columns = ['id', 'rating']


# In[21]:


movies_rating.head()


# In[22]:


movies_md['id'] = movies_md['id'].astype(int)


# In[23]:


movies_df = pd.merge(movies_md, keywords, on='id', how='left')


# In[24]:


movies_df.head()


# In[25]:


movies_df.reset_index(inplace=True, drop=True)


# In[26]:


movies_df = pd.merge(movies_df, movies_credits, on='id', how='left')
movies_df.reset_index(inplace=True, drop=True)


# In[27]:


movies_df.head()


# In[28]:


movies_df['genre'] = movies_df['genres'].apply(lambda x: [i['name'] for i in eval(x)])

movies_df.head()


# In[29]:


movies_df['genre'] = movies_df['genre'].apply(lambda x: [i.replace(" ", "") for i in x])

movies_df.head()


# In[30]:


movies_df.isnull().sum()


# In[31]:


movies_df['keywords'].fillna('[]', inplace=True)


# In[32]:


movies_df['genre'] = movies_df['genre'].apply(lambda x: ', '.join(x))


# In[33]:


# movies_df['genre'] = movies_df['genre'].apply(lambda x: ' '.join(x))


# In[34]:


movies_df.drop('genres', axis=1, inplace=True)


# In[35]:


movies_df.head()


# In[36]:


from collections import defaultdict

all_genres = defaultdict(int)

for genres in movies_df['genre']:
    for genre in genres.split(','):
        all_genres[genre.strip()] += 1


# In[38]:


from wordcloud import WordCloud

genres_cloud = WordCloud(width=800, height=400, background_color='white', colormap='gnuplot').generate_from_frequencies(all_genres)

plt.figure(figsize=(16,10))
plt.imshow(genres_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[39]:


movies_df['cast'] = movies_df['cast'].apply(lambda x: [i['name'] for i in eval(x)])
movies_df['cast'] = movies_df['cast'].apply(lambda x: ' '.join([i.replace(" ", "") for i in x]))


# In[40]:


movies_df['keywords'] = movies_df['keywords'].apply(lambda x: [i['name'] for i in eval(x)])
movies_df['keywords'] = movies_df['keywords'].apply(lambda x: ' '.join([i.replace(" ", "") for i in x]))


# In[41]:


movies_df.head()


# In[42]:


movies_df['tags'] = movies_df['overview']+' '+movies_df['keywords']+' '+movies_df['cast']+' '+movies_df['genre']+' '+movies_df['original_title']


# In[43]:


movies_df['tags']


# In[44]:


movies_df.drop(['genre', 'keywords', 'cast', 'overview'], axis=1, inplace=True)


# In[45]:


movies_df['runtime'] = movies_df['runtime'].astype("int")


# In[46]:


movies_df.head()


# In[47]:


movies_df.isnull().sum()


# In[48]:


movies_df.drop(movies_df[movies_df['tags'].isnull()].index, inplace=True)


# In[49]:


movies_df.shape


# In[50]:


movies_df.drop_duplicates(inplace=True)


# In[51]:


movies_df.shape


# In[65]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[66]:


# Common words have less IDF
# Unique Words have high IDF
tfidf = TfidfVectorizer(max_features=5000)


# In[67]:


vectorized_data = tfidf.fit_transform(movies_df['tags'].values)


# In[70]:


tfidf.get_feature_names()


# In[71]:


vectorized_dataframe = pd.DataFrame(vectorized_data.toarray(), index=movies_df['tags'].index.tolist())


# In[72]:


vectorized_dataframe.head()


# In[73]:


vectorized_dataframe.shape


# In[74]:


from sklearn.decomposition import TruncatedSVD


# In[75]:


svd = TruncatedSVD(n_components=3000)

reduced_data = svd.fit_transform(vectorized_dataframe)


# In[76]:


reduced_data.shape


# In[77]:


reduced_data


# In[78]:


svd.explained_variance_ratio_.cumsum()


# In[79]:


from sklearn.metrics.pairwise import cosine_similarity


# In[80]:


similarity = cosine_similarity(reduced_data)


# In[81]:


similarity


# In[82]:


def recomendation_system(movie):
    id_of_movie = movies_df[movies_df['original_title']==movie].index[0]
    distances = similarity[id_of_movie]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:20]
    for movie_id in movie_list:
        print(movies_df.iloc[movie_id[0]].original_title + " (" + movies_df.iloc[movie_id[0]].release_date + ")" +
              " : " + str(movies_df.iloc[movie_id[0]].runtime) + " Menit")


# In[83]:


recomendation_system('The Matrix')


# In[84]:


recomendation_system('Jumanji')


# In[90]:


from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
# from surprise.model_selection import KFold
reader = Reader()

data = Dataset.load_from_df(rating[['userId', 'movieId', 'rating']], reader)
# kf = KFold(n_splits=5)
# kf.split(data)


# In[91]:


Svd = SVD()
cross_validate(Svd, data, measures=['RMSE', 'MAE'], cv=5)


# In[100]:


trainset = data.build_full_trainset()
Svd.fit(trainset)


# In[105]:


rating[rating['userId'] == 1]


# In[107]:


Svd.predict(1, 302, 3)

