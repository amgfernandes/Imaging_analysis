#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


cd "D:/Projector measurements"


# In[3]:


'''Baseline correction function removes baseline absorption from an acquired spectrum'''
optoma_corrected=pd.read_csv('O119 behavioral setup optoma corrected.csv', sep=';',skiprows=33,
                 names=["nm","intensity"], decimal=",",skipfooter=1, engine='python')#skip rows with metadata and last row

Rhea_LGPA72G_corrected=pd.read_csv('Rhea LG PA72G OD1 corrected.csv', sep=';',skiprows=33,
                 names=["nm","intensity"], decimal=",",skipfooter=1, engine='python')#skip rows with metadata and last row

Rhea_lightcrafter_corrected=pd.read_csv('Rhea lightcrafter wratten corrected.csv', sep=';',skiprows=33,
                 names=["nm","intensity"], decimal=",",skipfooter=1, engine='python')#skip rows with metadata and last row


# In[4]:


optoma_uncorrected=pd.read_csv('O119 behavioral setup optoma uncorrected.csv', sep=';',skiprows=33,
                 names=["nm","intensity"], decimal=",",skipfooter=1, engine='python')#skip rows with metadata and last row

Rhea_LGPA72G_uncorrected=pd.read_csv('Rhea LG PA72G OD1 uncorrected.csv', sep=';',skiprows=33,
                 names=["nm","intensity"], decimal=",",skipfooter=1, engine='python')#skip rows with metadata and last row

Rhea_lightcrafter_uncorrected=pd.read_csv('Rhea lightcrafter wratten uncorrected.csv', sep=';',skiprows=33,
                 names=["nm","intensity"], decimal=",",skipfooter=1, engine='python')#skip rows with metadata and last row


# In[5]:


projectors=[optoma_corrected,Rhea_LGPA72G_corrected,Rhea_lightcrafter_corrected,
            optoma_uncorrected,Rhea_LGPA72G_uncorrected,Rhea_lightcrafter_uncorrected]
projectors_names=['Optoma_corrected','Rhea_LGPA72G_corrected','Rhea_lightcrafter_corrected',
                  'Optoma_uncorrected','Rhea_LGPA72G_uncorrected','Rhea_lightcrafter_uncorrected']


# In[6]:


count=-1
colors=['m','g','b','k','c','r']
for projector in projectors:
    count+=1
    #plt.figure(figsize=(5,5))
    plt.plot(projector.nm[700:-350],projector.intensity[700:-350], color=colors[count])
    plt.xlabel('Wavelenght')
    plt.ylabel('Intensity')
    #plt.title(projectors_names[count])
    sns.despine()


# In[7]:


count=-1
colors=['m','g','b','k','c','r']
for projector in projectors:
    count+=1
    plt.figure(figsize=(5,5))
    plt.plot(projector.nm[700:-350],projector.intensity[700:-350], color=colors[count])
    plt.xlabel('Wavelenght')
    plt.ylabel('Intensity')
    plt.title(projectors_names[count])
    sns.despine()
    plt.savefig(str(projectors_names[count])+'.png')
    plt.savefig(str(projectors_names[count])+'.svg')


# In[ ]:




