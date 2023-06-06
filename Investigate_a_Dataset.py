#!/usr/bin/env python
# coding: utf-8

# 
# 
# # Project: Investigate No-show appointments Dataset
# 
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='Introduction'></a>
# ## Introduction
# 
# This dataset collects information from 100k medical appointments in Brazil and is focused on the question of whether or not patients show up for their appointment. A number of characteristics about the patient are included in each row.
# 
# ● ‘ScheduledDay’ tells us on what day the patient set up their appointment.
# ● ‘Neighborhood’ indicates the location of the hospital.
# ● ‘Scholarship’ indicates whether or not the patient is enrolled in Brasilian welfare program Bolsa Família.
# ● Be careful about the encoding of the last column: it says ‘No’ if the patient showed up to their appointment, and ‘Yes’ if they did not show up 
# <a id='Questions'></a>
# ## Questions
# 
# We will try to understand what factors are important for us to know in order to predict if apatient will show up for there scheduled appointment?

# In[68]:


import numpy as np
import pandas as pd
from scipy import stats
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import random
get_ipython().run_line_magic('matplotlib', 'inline')


# In[69]:


# Upgrade pandas to use dataframe.explode() function. 
get_ipython().system('pip install --upgrade pandas==0.25.0')


# <a id='wrangling'></a>
# ## Data Wrangling
# 

# In[71]:


df= pd.read_csv('noshowappointments-kagglev2-may-2016.csv')
df.head()


# In[72]:


# Returns the percentage of male and female who visited the 
# hospital on their appointment day with their Age
def people_visited(age_group, gender):
    grouped_by_total = clean_appointment_data.groupby(['age_group', 'Gender']).size()[age_group,gender].astype('float')
    grouped_by_visiting_gender =         clean_appointment_data.groupby(['age_group', 'people_showed_up', 'Gender']).size()[age_group,1,gender].astype('float')
    visited_gender_pct = (grouped_by_visiting_gender / grouped_by_total * 100).round(2)
    
    return visited_gender_pct


# In[73]:


df.shape


# In[74]:


df.info()


# No missing data

# In[75]:


df.describe()


# ● The Mean of ages is 37 Years
# ● Maximum age is 115 years
# ● There is probably a mistak with one of the patient's age.it shows-1 year,which dosn't make sense.

# 
# ### Data Cleaning
#  

# In[76]:


df.drop(['PatientId', 'AppointmentID', 'ScheduledDay', 'AppointmentDay'], axis = 1, inplace=True)
df.head()


# In[77]:


df.rename(columns={'No-show': 'No_show'}, inplace=True)
df.head()


# In[78]:


df.rename(columns={'Hipertension' : 'Hypertension'}, inplace=True)
df.head()


#  Exploratory Data Analysis
# 
# Lets take a look on the data

# In[79]:


df.hist(figsize= (15,12));


# ● Most of the patients didn't suffer from chronic diseases nor are handicapped
# ● About 18 % (20000 out of 110000 )suffered from hypertension
# ● Number of patients who received an SMS is half the number of those who did not
# ● About 9% are enrolled in the brasilian welfare program

# Before anything let's see how many patients attended

# In[80]:


show = df.No_show == 'No'
noshow = df.No_show == 'Yes'


# In[81]:


df[show].count()


# In[82]:


df[noshow].count()


# The number of those who showed at the clinic was about 4 times those who did not show

# In[83]:


plt.figure(figsize=[14.70, 8.27])
df.Age[show].hist(alpha=0.5, label='show')
df.Age[noshow].hist(alpha=0.5, label='noshow')
plt.legend()
plt.title('comparison between who showed to who did not according to Age')
plt.xlabel('Age')
plt.ylabel('Patients Number');


# Patients in the age group 0-10 showed more than all the other age groups, followed by the age group 35-70.
# The older they get,the less they tend to get an appointment
# from the above visualization it appers that the ratio of show to noshow is nearly the same for all ages except for 'Age 0' and 'Age 1' we will get better clarity on the ratio of show to noshow for all ages . so age does not affect the commitment to visit much

# In[84]:


plt.figure(figsize=[14.70, 8.27])
df.Neighbourhood[show].value_counts().plot(kind='bar', alpha=0.5, color='blue', label='show')
df.Neighbourhood[noshow].value_counts().plot(kind='bar', alpha=0.5, color='orange', label='noshow')
plt.legend()
plt.title('comparison between who showed to who did not according to Neighbourhood')
plt.xlabel('Neighbourhood')
plt.ylabel('Patients Number');


# It seems that neighbourhood is strongly affecting the showing of patients at the clinic.and we can see that the number of patient for few neighbourhood's is very high 

# In[85]:


plt.figure(figsize=[14.70, 8.27])
df.Gender[show].hist(alpha=0.5, label='show')
df.Gender[noshow].hist(alpha=0.5, label='noshow')
plt.legend()
plt.title('comparison between who showed to who did not according to Gender')
plt.xlabel('Gender')
plt.ylabel('Patients Number');


# In[86]:


print(df.Scholarship[show].value_counts())
print(df.Scholarship[noshow].value_counts())


# Being enrolled in the brasilian welfare program is insignificant

# We can see that of the 88,000 patients that appeared, about 57,000 were female and 31,000 were male of the 22,500 patients who did not com for a visit
# about 15,000 were females and 7,500 were males the ratio of females to males who attended appears to be the same as that which did not com to visit and therefore gender does not affect
# for female 26%
# for male 24%

# In[87]:


plt.figure(figsize=[14.70, 8.27])
df.Hypertension[show].hist(alpha=0.5, label='show')
df.Hypertension[noshow].hist(alpha=0.5, label='noshow')
plt.legend()
plt.title('comparison between who showed to who did not according to Hypertension')
plt.xlabel('Hypertension')
plt.ylabel('Patients Number');


# Hypertension is insignificant, we can see that there are about 88,000 patients suffering from high blood pressure and about
# 78% of them attended the visit.Of 21801 patients with no high blood pressure, about 85% came to visit. therefore the high blood pressure feature can help us determine whether a patient will show up on a post-appointment visit

# In[88]:


plt.figure(figsize=[14.70, 8.27])
df.Diabetes[show].hist(alpha=0.5, label='show')
df.Diabetes[noshow].hist(alpha=0.5, label='noshow')
plt.legend()
plt.title('comparison between who showed to who did not according to Diabetes')
plt.xlabel('Diabetes')
plt.ylabel('Patients Number');


# Diabetes is insignificant. We can see that there are about 102,000 Diabetics and about 80% of them attend the visit. Of the 7943 diabetic patient.about 83% came to visit, the diabetes feature can help us determine whether a patient will attend the post-appointment vist

# In[89]:


plt.figure(figsize=[14.70, 8.27])
df.Alcoholism[show].hist(alpha=0.5, label='show')
df.Alcoholism[noshow].hist(alpha=0.5, label='noshow')
plt.legend()
plt.title('comparison between who showed to who did not according to Alcoholism')
plt.xlabel('Alcoholism')
plt.ylabel('Patients Number');


# Alcoholism is insignificant.We can see that there are about 107,000 patients who do not suffer from alcholism and about 80% of them attend the visit.of the 3360 patients with alcohol addiction about 80% attended the visit .since the rate of visits for non-alcoholic patients is the same this may not help us determine whether or not the patient is coming for a visit

# In[90]:


plt.figure(figsize=[14.70, 8.27])
df.Handcap[show].hist(alpha=0.5, label='show')
df.Handcap[noshow].hist(alpha=0.5, label='noshow')
plt.legend()
plt.title('comparison between who showed to who did not according to Handcap')
plt.xlabel('Handcap')
plt.ylabel('Patients Number');


# Handcap is insignificant

# In[91]:


plt.figure(figsize=[14.70, 8.27])
df.SMS_received[show].hist(alpha=0.5, label='show')
df.SMS_received[noshow].hist(alpha=0.5, label='noshow')
plt.legend()
plt.title('comparison between who showed to who did not according to SMS_received')
plt.xlabel('SMS_received')
plt.ylabel('Patients Number');


# it's a bit strange to see that more people showed without receiving an SMS!

# 
# ## Conclusions
#  at the end, Neighbourhood is strongly related to the patients showing up at the clinic.
#  
#  Age also has it's role as in the 0-10 age group were the most to show up followed by age group 35-70
#  
#  although more people showed without receiving an SMS!
# 
# 
# ### Limitations
# could not detect direct corrlation between patients showing no_showing and many characterstics such as gender, chronic diseases, disabilities
# 

# In[92]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])

