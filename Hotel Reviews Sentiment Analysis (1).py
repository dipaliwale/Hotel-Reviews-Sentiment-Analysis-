#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Project(Hotel Reviews Sentiment Analysis)


# In[2]:


import numpy as np
import pandas as pd


# In[3]:


df=pd.read_csv("C:\\Users\\DIPALI\\Desktop\\Desktop\\Data Analyst class 2023\\hotel.csv")
df.head()


# In[4]:


df['Review'][0]


# In[5]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[6]:


#SentimentIntensityAnalyser ==> NLTK(Natural Language Tool kit) ==>text ==>Review(Positive,Negative,Neutral)


# In[24]:


ratings=df['Rating'].value_counts()
ratings


# In[8]:


numbers=ratings.index
numbers


# In[9]:


quantity=ratings.values
quantity


# In[10]:


custom_colors=['skyblue','yellowgreen','tomato','blue','red']
plt.figure(figsize=(7,7))
plt.pie(quantity,labels=numbers,colors=custom_colors)
central_circle=plt.Circle((0,0),0.5,color='white')
fig=plt.gcf()
fig.gca().add_artist(central_circle)
plt.rc('font',size=12)
plt.title('Hotel Reviews Rating',fontsize=20)
plt.show()


# In[20]:


import nltk
nltk.download('vader_lexicon')   


# In[ ]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer


# In[21]:


sentiments = SentimentIntensityAnalyzer()


# In[15]:


# To analyze the sentiments of the hotel reviews we will add 3 additional coloum to 
#this dataset as Positive, Negative ,and Neutral by calculating the sentiment scores of the reviews.


# In[22]:


df['Positive']=[sentiments.polarity_scores(i)['pos'] for i in df['Review']]
df['Negative']=[sentiments.polarity_scores(i)['neg'] for i in df['Review']]
df['Neutral']=[sentiments.polarity_scores(i)['neu'] for i in df['Review']]


# In[23]:


df.head()


# In[25]:


#According to the reviews, hotel guests seem satisfied with the services,now lets take a look at 
# how most people think about hotel services based on sentiments of the reviews.


# In[26]:


x=sum(df['Positive'])
y=sum(df['Negative'])
z=sum(df['Neutral'])


# In[28]:


print("Total Positive =",x)
print("Total Negative =",y)
print("Total Neutral =",z)


# In[29]:


def sentiment_scores(a,b,c):
    if(a>b) and (a>c):
        print("Positive😊")
    elif(b>a) and (b>c):
        print("Negative😊")
    else:
        print("Neutral")
sentiment_scores(x,y,z)


# In[ ]:




