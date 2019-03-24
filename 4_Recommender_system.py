
#commands to execute. e.g. python 4_Recommender_system.py -c 762616
#commands to execute. e.g. python 4_Recommender_system.py -u 'Srinivas'

import udemy_functions # all functions to build the clusters and the recomender systems
import scipy.stats as st 
import pandas as pd
import numpy as np
import ast
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from nltk.corpus import stopwords
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('foo', type=udemy_functions.recommend_for_user)
parser = argparse.ArgumentParser(description='Recommender syste mfor users or courses')
parser.add_argument('-c', '--course_id', help='ID of the taken courses',  type=int)
parser.add_argument('-u', '--user_name', help='Name of the user to recommend new courses')
args = vars(parser.parse_args())
#action='store_const',

#import data
df_courses=pd.read_csv('data/cleaned/df_courses.csv', index_col=0, sep=' ', converters={"objectives": ast.literal_eval})
df_courses.head()

df_reviews=pd.read_csv('data/cleaned/df_reviews.csv', index_col=0)
df_reviews.head()

#preparation of attribute description for clustering
vocab_frame_descr=udemy_functions.vocab_stem(df_courses['description_text'])
vocab_frame_descr=udemy_functions.drop_words(vocab_frame_descr)
StopWords=set(stopwords.words('english')+list(punctuation)+["’", "n't", "'s", "--", "-", "...", "``", "''", "“", "039"])

vectorizer_descr= TfidfVectorizer(stop_words=StopWords, tokenizer=udemy_functions.tokenize, max_features=1000, max_df=0.8)
X_descr=vectorizer_descr.fit_transform(df_courses['description_text'])
word_features_descr = vectorizer_descr.get_feature_names()

#clustering algorithm
model_kmeans=pickle.load(open('kmeans8.sav', 'rb')) 
df_courses['cluster_descr']=model_kmeans.predict(X_descr)

#preparation for the recommender sytsem
rel_cols=['avg_rating',  'has_certificate',  'instructional_level', 'num_lectures','num_quizzes',
          'num_practice_tests','is_practice_test_course', 'num_article_assets', 'num_curriculum_items',
          'num_subscribers','num_reviews',  'price', 'primary_subcategory','cluster_descr']
df_rel=df_courses[rel_cols]
df_rel['has_certificate']=df_rel['has_certificate'].astype(int)
df_rel['cluster_descr']=df_rel['cluster_descr'].astype(str)
dummies=pd.get_dummies(df_rel[['primary_subcategory', 'instructional_level','cluster_descr']], prefix=['subcat', 'level', 'cluster'])
df_rel.drop(columns=['primary_subcategory', 'instructional_level', 'cluster_descr'], inplace=True)
df_rel=pd.concat([df_rel,dummies], axis=1)
df_norm=udemy_functions.normalize_features(df_rel)

#recommender system

if args['course_id']!=None:
	course_id_arg=args['course_id']
	print(udemy_functions.recommend_courses(args['course_id'], 5, df_courses, df_norm))

if args['user_name']!=None:
	print(udemy_functions.recommend_for_user(args['user_name'], 5,df_reviews, df_courses, df_norm))
