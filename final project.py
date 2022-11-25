#!/usr/bin/env python
# coding: utf-8

# # Anime dataset
# ## introduction
# About Dataset
# Context
# This dataset contains informations about Anime (16k), Reviews (130k) and Profiles (47k) crawled from https://myanimelist.net/.
# 
# The dataset contains 3 files:
# 
# animes.csv contains list of anime, with title, title synonyms, genre, duration, rank, populatiry, score, airing date, episodes and many other important data about individual anime providing sufficient information about trends in time about important aspects of anime. Rank is in float format in csv, but it contains only integer value. This is due to NaN values and their representation in pandas.
# 
# reviews.csv contains information about reviews users x animes, with text review and scores.

# In[242]:


#importing the required libraries
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[139]:


pip install plotting


# In[140]:


get_ipython().system('pip install --upgrade pandas')


# we are gonna use 2 datasets here. one is amines dataset
# second is reviews dataset which has reviews given by various people to the anime we are gonna clean those and merge them to one for analysis

# In[141]:


#importing the animes file
file_path = 'E:/downloads/archive (3)/animes.csv'

dataset = pd.read_csv(file_path)


# In[142]:


#to copy the dataset for subdatasets
animes_df = dataset.copy()


# In[143]:


#importing the reviews file
file_path = 'E:/downloads/archive (3)/reviews.csv'

dataset = pd.read_csv(file_path)


# In[144]:


reviews_df = dataset.copy()


# In[145]:


animes_df.head()


# In[146]:


animes_df.tail()


# In[147]:


reviews_df.head()


# In[148]:


reviews_df.tail()


# now are gonna learn more about the columns in each dataset 
# first we go with animes_df which has the data about the animes 

# In[149]:


# display the coloumns
list(animes_df.columns)


# In[150]:


#display no.of rows aand coloumns
numberofrows_1, numberofcolumns_1 = animes_df.shape
print('There are {} rows and {} columns'.format(numberofrows_1, numberofcolumns_1)+' in our Animes Dataset.')


# In[151]:


#displays the type of coloumns
animes_df.info()


# In[152]:


animes_df.describe()


# ## synopsis from above data
# animes.csv contains list of anime, with title, title synonyms, genre, duration, rank, populatiry, score, airing date, episodes and many other important data about individual anime providing sufficient information about trends in time about important aspects of anime. Rank is in float format in csv, but it contains only integer value. This is due to NaN values and their representation in pandas.
# 1. uid        = this is the representation of the uid's given to each anime
# 2. title      = this is the title of the anime
# 3. synopsis   = synopsis of the given anime i.e small explanation
# 4. genre      = shows what type of the anime it is
# 5. aired      = it is the time of when the anime is released
# 6. episodes   = no.of episodes animated in that anime
# 7. members    = no.of members worked on that anime
# 8. popularity = its the numerical represention of the popularity of the anime
# 9. ranked     = its the ranking of the anime
# 10. score     = score given to the anime
# 11. img_url   = url of the image in the website
# 12. link      = link to the anime page in myanimelist.com

# In[153]:


#shows us the different types of ranks without repetition
animes_df["ranked"].unique()


# In[154]:


#shows us how many different types of ranks without repetition
len(animes_df["ranked"].unique())


# In[155]:


#shows us the different types of genre's without repetition
animes_df["genre"].unique()


# In[156]:


#shows us how many different types of genre's without repetition
len(animes_df["genre"].unique())


# In[157]:


#can be used to find the no of rows 
animes_df["uid"].count()


# now are gonna learn more about the columns in each dataset 
# first we go with reviews_df which has the data about the animes 

# In[158]:


# display the coloumns
list(reviews_df.columns)


# In[159]:


#display no.of rows aand coloumns
numberofrows_1, numberofcolumns_1 = reviews_df.shape
print('There are {} rows and {} columns'.format(numberofrows_1, numberofcolumns_1)+' in our Reviews Dataset.')


# In[160]:


#displays the type of coloumns
reviews_df.info()


# In[161]:


reviews_df.describe()


# ## synopsis from above data
# reviews.csv contains list of anime, with uid, anime_uid, text, profile, text, score, scores, link. important data about individual anime providing sufficient information about what are animes that the prople like and their scores on the anime.
# 1. uid        = this is the representation of the uid's given to each anime
# 2. profile    = this is the name of the person
# 3. anime_uid  = this is the uid given to anime
# 4. text       = we can see that this is just a null column which should be removed
# 5. score      = integer score of the anime given
# 6. scores     = scores given to the anime on different aspects
# 7. link       = link to the anime page in myanimelist.com

# In[162]:


reviews_df["score"].unique()


# In[163]:


len(reviews_df["score"].unique())


# In[164]:


reviews_df["scores"].head()


# In[165]:


##Finding the corelation.
corrmat = animes_df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True, cmap= "YlGnBu");
plt.show()


# In[166]:


##Finding the corelation.
corrmat = reviews_df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True, cmap= "YlGnBu");
plt.show()


# the correlation done here is not a good representation of the dataset since most of the columns need to formatted in a readable way and many need to be removed and some to be cleaned.
# first we go with the null removal of the datasets.
# 
# # Null Removal of animes_df

# In[167]:


#animes_df null removal
animes_df.isnull()


# In[168]:


animes_df.isnull().sum()


# In[169]:


meanVal=animes_df['episodes'].mean()
animes_df['episodes'].fillna(value=meanVal, inplace=True)
animes_df.isnull().sum()


# In[170]:


meanVal=animes_df['ranked'].mean()
animes_df['ranked'].fillna(value=meanVal, inplace=True)
animes_df.isnull().sum()


# In[171]:


animes_df['score'] = animes_df['score'].fillna(animes_df['score'].mode()[0])
animes_df.isnull().sum()


# In[172]:


animes_df1=animes_df.replace(np.nan, 'NaN', regex=True)
animes_df.isnull().sum()


# In[173]:


animes_df1.isnull().sum()


# # Null Removal of reviews_df

# In[174]:


#reviews_df null removal
reviews_df.isnull()


# In[175]:


reviews_df.isnull().sum()


# # this dataset has now null balues but the "scores" column which is in the fomal of object has multiple scores in a single column we can clean this column and and make it suitable for many analysis

# In[176]:


reviews_df["scores"].head()


# here we see that the scores column which is a object datatype has the scores which is given. it is catogarized into 6 types which are                               
# 1. Overall                                          
# 2. Story
# 3. Animation
# 4. Sound
# 5. Character
# 6. Enjoyment
# 
# 
# now we have to distinguish them into their own columns where they are given their respective score.this is a part of data cleaning which can used for further assitance in data visualization

# In[177]:


#since each score is differentiated by "," we can use this to split the single column into various different columns
reviews_df['overall']=reviews_df.scores.apply(lambda x: x.split(",")[0])


# In[178]:


reviews_df.head()


# we see here that we succcesfully able to spilt the scores column. now we have to do same for every type of score

# In[179]:


reviews_df['story']=reviews_df.scores.apply(lambda x: x.split(",")[1])


# In[180]:


reviews_df['animation']=reviews_df.scores.apply(lambda x: x.split(",")[2])


# In[181]:


reviews_df['sound']=reviews_df.scores.apply(lambda x: x.split(",")[3])


# In[182]:


reviews_df['character']=reviews_df.scores.apply(lambda x: x.split(",")[4])


# In[183]:


reviews_df['enjoyment']=reviews_df.scores.apply(lambda x: x.split(",")[5])


# In[184]:


reviews_df.head()


# all the required categories are distinguished. now we have to elimainate the characters of aA-zZ and all other symbols in the strings which is required for futher data visualization

# In[185]:


#this codeline rewrites every string from aA-zZ and symbols in the column to the assinged value which is null here
reviews_df['overall'] = reviews_df['overall'].str.replace(r'\D', '')


# In[186]:


reviews_df['story'] = reviews_df['story'].str.replace(r'\D', '')


# In[187]:


reviews_df['animation'] = reviews_df['animation'].str.replace(r'\D', '')


# In[188]:


reviews_df['sound'] = reviews_df['sound'].str.replace(r'\D', '')


# In[189]:


reviews_df['character'] = reviews_df['character'].str.replace(r'\D', '')


# In[190]:


reviews_df['enjoyment'] = reviews_df['enjoyment'].str.replace(r'\D', '')


# In[191]:


reviews_df.head()


# In[192]:


reviews_df.info()


# here the new columns are of object datatype but we need them to be of int datatype to be able to do the data analysis

# In[193]:


reviews_df['overall'] = pd.to_numeric(reviews_df['overall'])


# In[194]:


reviews_df['story'] = pd.to_numeric(reviews_df['story'])


# In[195]:


reviews_df['animation'] = pd.to_numeric(reviews_df['animation'])


# In[196]:


reviews_df['sound'] = pd.to_numeric(reviews_df['sound'])


# In[197]:


reviews_df['character'] = pd.to_numeric(reviews_df['character'])


# In[198]:


reviews_df['enjoyment'] = pd.to_numeric(reviews_df['enjoyment'])


# In[199]:


reviews_df.info()


# # merging both datasets
# now we have to merge the both datasets for futher analysis. wew can do this using merge function but we need to delete the columns that are no more required and also there are multiple columns with score name and we have to chose one fo them for keeping

# In[200]:


#deleting the not required columns from each dataset
animes_df1.info()


# here img_url,link are not required for analysis

# In[201]:


animes_df1.drop("img_url",axis=1,inplace=True)


# In[202]:


animes_df1.drop("link",axis=1,inplace=True)


# In[203]:


animes_df1.info()


# now the same for the reviews dataset
# 

# In[204]:


reviews_df.info()


# now here we know that text is null column. and we also dont need link in the dataset

# In[205]:


reviews_df.drop("text",axis=1,inplace=True)


# In[206]:


reviews_df.drop("scores",axis=1,inplace=True)


# In[207]:


reviews_df.drop("link",axis=1,inplace=True)


# In[208]:


reviews_df.info()


# In[209]:


animes_df1['score']


# In[210]:


animes_df1["score"].unique()


# In[211]:


reviews_df['score']


# In[212]:


reviews_df["score"].unique()


# here we see that the score column of reviews dataset is more organised and in a usable format for data analysis

# In[213]:


animes_df1.drop("score",axis=1,inplace=True)


# In[214]:


animes_df1.head()


# In[215]:


reviews_df.head()


# merging of the datasets

# In[216]:


anime = pd.merge(animes_df1, reviews_df, how='inner',on=['uid']) 


# In[217]:


anime.head()


# In[218]:


anime.info()


# In[219]:


numberofrows_1, numberofcolumns_1 = anime.shape
print('There are {} rows and {} columns'.format(numberofrows_1, numberofcolumns_1)+' in our Reviews Dataset.')	


# the loss of the rows is due to the rows having no similarity between the datasets on basis of uid

# # univariate analysis

# In[220]:


anime['score'].hist()


# In[221]:


anime['ranked'].hist()


# In[222]:


plt.scatter(anime.index,anime['score'])
plt.show()


# In[223]:


plt.scatter(anime.index,anime['ranked'])
plt.show()


# In[225]:


plt.scatter(anime.index,anime['episodes'])
plt.show()3


# In[227]:


plt.scatter(anime.index,anime['popularity'])
plt.show()


# In[230]:


anime.overall.value_counts(normalize=True)


# In[233]:


anime.overall.value_counts(normalize=True).plot.bar()


# since the scores assinged to the animes are of similar tyoe it will be hard to use those for data analysis so, we can classiy the overall score column to create a better understanding

# In[249]:


def overall_rating(overall):
    if overall >= 8:
        return 'good'
    elif overall >= 6:
        return 'average'
    elif overall >= 4:
        return 'bellow average'
    else:
        return 'bad'


# In[253]:


anime['overall_rating']= anime['overall'].apply(overall_rating)


# In[254]:


anime.head()


# In[256]:


anime["overall_rating"].unique()


# In[258]:


len(anime["overall_rating"].unique())


# In[259]:


anime.overall_rating.value_counts().plot.pie()


# In[260]:


plt.figure(figsize=(5,5))
plt.pie(anime["overall_rating"].value_counts(), startangle=90,autopct='%.2f',labels=['good','average','bellow average','bad'],shadow=True)
plt.title("overall_rating according to percentage")
plt.show()


# ## bivariate analysis

# In[239]:


#Finding the corelation.
corrmat = anime.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True, cmap= "YlGnBu");
plt.show()


# In[241]:


sns.heatmap(anime.corr(), cmap="Greens")


# In[246]:


pd.plotting.scatter_matrix(anime, alpha=0.2)


# ## statistical analysis

# In[248]:


anime.describe()


# 
