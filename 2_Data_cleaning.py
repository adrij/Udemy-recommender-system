###########################################################################
# Through the data cleaning process I did the following operations on the raw dataset:
# 1. import the raw data
# 2. transform the relevant columns
# 3. filter the dataset
# 4. keep only the relevant columns
# 5. drop the duplicates
# 6. treat the missing values
# 7. save the cleaned data

###########################################################################
# 0. import relevant packages
#import the relevant packages
import pandas as pd
import numpy as np
import datetime
import scipy.stats as st
import ast
import re 
import matplotlib.pyplot as plt
import udemy_functions

import warnings
warnings.filterwarnings("ignore")

##################################################
###########Clean the courses data#################
##################################################

# 1. import the raw file
df=pd.read_csv('data/raw/df_courses.csv', index_col=0)

# 2. transform the relevant columns
df['primary_category']=udemy_functions.transform_col(df['primary_category'], 'title')
df['primary_subcategory']=udemy_functions.transform_col(df['primary_subcategory'], 'title')
df['content_info']=df['content_info'].apply(udemy_functions.get_float)
df['price']=df['price'].apply(udemy_functions.get_float)
df['published_time']=pd.to_datetime(df['published_time'])
df['published_since_month']=(datetime.datetime.now()-df['published_time']).apply(lambda x: int(x.days/30))
df['objectives']=udemy_functions.transform_col(df['objectives'])
df['description_text']=df['description'].apply(udemy_functions.remove_tags)

#transform the rating distribution 
rating_orig=[]
rating_rel=[]
for i, rating in enumerate(df['rating_distribution'].values):
    total=0
    temp={}
    temp_rel={}
    if rating:
        rating=ast.literal_eval(rating)
        for rating_j in rating:
            j=rating_j['rating']
            count_j=rating_j['count']
            total+=count_j
            temp[j]=count_j
        rating_orig.append(temp)
        if total>0:
            for k,v in temp.items():
                temp_rel[k]=round(v*1.0/total,3)
            rating_rel.append(temp_rel)
        else:
            rating_rel.append({1:0, 2:0, 3:0, 4:0, 5:0})
    else:
        rating_rel.append({1:0, 2:0, 3:0, 4:0, 5:0})
        rating_orig.append({1:0, 2:0, 3:0, 4:0, 5:0})
df_rating=pd.DataFrame(rating_rel)
df_rating.columns=['rating_1', 'rating_2', 'rating_3', 'rating_4','rating_5']
df=pd.concat([df, df_rating], axis=1)

# 3. filter the dataset
df=df[(df['is_published']== True ) & (df['status_label']== 'Live')]
#drop the columns that are transformed or not relevant any more
df.drop(columns=['published_time','rating_distribution','status_label', 'is_published', 'rating', 'description' ], axis=1, inplace=True)


# 4. keep the relevant columns
cols=['avg_rating', 'avg_rating_recent', 'description_text', 'has_certificate',  'is_paid',
      'id', 'instructional_level', 'is_enrollable_on_mobile', 'is_owned_by_instructor_team', 'is_practice_test_course', 
      'num_article_assets' , 'num_curriculum_items','num_lectures', 'num_practice_tests', 'num_quizzes',
      'num_subscribers', 'num_reviews', 'objectives', 'price','published_title', 'relevancy_score','rating_1', 
      'rating_2', 'rating_3', 'rating_4','rating_5', 'published_since_month', 'primary_category', 'primary_subcategory' ]
df=df[cols]

# 5. drop the duplicates
df=df.drop_duplicates(subset='id', keep='first')

# 6. check the missing values
#The free courses are labeled as free -> change price for these courses: 0 
df['price']=df['price'].fillna(0)

#drop the missings
df.dropna(how='any', inplace=True)

#in the objectives, there are empty lists 
index_to_drop=df[df['objectives'].apply(lambda x: x==list([]))].index
df.drop(index=index_to_drop, inplace=True)

# 7. save the cleaned dataset
df.to_csv('data/cleaned/df_courses.csv', sep=' ')

##################################################
###########Clean the reviews data#################
##################################################

# 1. import the raw data
df_review=pd.read_csv('data/raw/df_review.csv', index_col=0)

# 2. transform the relevant columns
df_review['user_name']=udemy_functions.transform_col(df_review['user'], 'display_name')
df_review['user_title']=udemy_functions.transform_col(df_review['user'], 'title')

# 3. filter the dataset from anonymized users (3 types)

df_review=df_review[~df_review['user_name'].isin(['Anonymized User', 'Private Udemy For Business User', 'Udemy User'])]

# 4. keep only the relevant columns
cols=['course_id', 'created', 'rating', 'user_name']
df_review=df_review[cols]

# 5. drop the duplicates
#the user names in the reviews data are not unique, it is impossible to build a recommender system based on the user ratings
df_review.drop_duplicates(inplace=True)

# 6. treat the missing values
#no missing values
# 7. save the cleaned data
df_review.to_csv('data/cleaned/df_reviews.csv')
