#Import allrelevant libraries
import pandas as pd
import numpy as np

import scipy.stats as st

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import requests
import os

import udemy_functions


####################### 1.1. Import the courses ####################
#Import the Business courses from the Udemy API  - limit is 10.000 

username='SirqldYB1Eg3dSaSlzNlr37Ix4wdIM1DELBPvfK6'
pw = os.environ['udemy_password']

list_json=[]
url='https://www.udemy.com/api-2.0/courses/?fields[course]=@all&page=1&category=Business'
i=0

while url!='':
    try:
        data_json=udemy_functions.get_data(url, username, pw)
        url=data_json['next']
        list_json.extend(data_json['results'])
        if i%50==0:
            print(i)
        i=+1
    except:
        continue

#Save the result in a dataframe and export to csv file
df_courses = pd.DataFrame.from_dict(list_json)
df_courses.to_csv('/data/raw/df_courses_.csv')


####################### 1.2. Import the reviews ####################
#For these courses, I downloaded the available reviews. The maximum number of available reviews for a course is 10.000.

for j, id_ in enumerate(df_courses['id'].values):    
    url="https://www.udemy.com/api-2.0/courses/{}/reviews/?page=1&page_size=100".format(id_)
    list_json_review=[]
    #for i in range(1,100):
    while url!='':
        try:
        	data_json=udemy_functions.get_data(url, username, pw)
        	url=data_json['next']
        	list_json_review.extend(data_json['results'])
        except:
        	continue
    if j==0:
        df_review= pd.DataFrame.from_dict(list_json_review)
        df_review['course_id']=id_
    else:
        df_review_unique = pd.DataFrame.from_dict(list_json_review)
        df_review_unique['course_id']=id_
        df_review = pd.concat([df_review, df_review_unique])
    if j%50==0:
            print(j)

#export to csv file        
df_review.to_csv('data/raw/df_review_.csv')
