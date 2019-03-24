# Udemy recommender system  

## Introduction

The following project is part of the [K2 Data Science](http://www.k2datascience.com/) bootcamp where I am currently enrolled. In Chapter 3 students are asked to build a data mining algorithm. I chose to implement a clustering algorithm and a recommender system.

[Udemy.com](https://www.udemy.com/) is an online learning platform with more than 100.000 courses and over 30 million students all over the world. The platform offers courses in different categories e.g. Business, Design or Marketing. 
With all the available options it is very hard to choose the proper course, since everyone has a different taste. A recommender system helps students choose the next course, without spending hours reading different course descriptions. It does not only spare time for the user, but helps to find something interesting based on their previous course choices.

My project is built up the following way:
At first, I clusterize the Business courses based on their description by means of NLP (natural language processing) techniques. 
Secondly, based on the new clusters and other relevant features (e.g. price or length of course) I build a recommender system.

## Structure

The project consists of 4 main parts: 
1. 1_Import_data.py: Getting the data through Udemy API
2. 2_Data_cleaning.py: Cleaning the data
3. 3_EDA.ipynb: Exploratory Data Analysis
4. 4_Clustering_and_RS.ipynb: Clustering and recommender system

Other files of the project:\
5. 4_Recommender_system.py: The relevant commands to run the recommender system\
6. kmeans8.sav : the final clustering algorithm exported\
7. Udemy_funcitons.py: all the functions which are needed for the projects

The first two parts are saved in python files, and the last two parts - where the visualization is also important - are saved in Jupyter notebooks. For the clustering and recommender system part I also created a script, which only contains the relevant commands: it prepares the data and imports the clustering algorithm (‘kmeans8.sav’). Thanks to the argparse library it is possible to run the recommender system from the command line: either with the parameter course_id (-c) or user_name (-u).

The functions of the jupyter notebooks are stored in the udemy_functions.py script.

## Approach

### Step 1: Organizing the Data

Udemy has an API, where all the course informations and user ratings can be achieved. I downloaded 10.000 Business courses through the API and loaded them in a Dataframe. (The limit is 10.000 records per query). For all of these courses I also downloaded the available user ratings. 

Name of dataframe | df_courses | df_reviews 
----------------- | -----------| -----------
Number of rows originally | 10.000 | 1.415.734
Number of rows after cleaning | 8.834 | 1.391.194
Final number of columns | 30 | 4

### Step 2: Cleaning the data

I followed these steps while cleaning the data:
* import the raw data
* transform the relevant columns
* filter the dataset
* keep the relevant columns
* drop duplicates
* treat the missing values
* save the cleaned data

The dataset was filtered so that it only contains live and public courses. After removing the duplicates and missing values, the cleaned dataset for courses (df_courses) consisted of 8.834 courses, which were analysed during the exploratory data analysis (EDA) part. The non-relevant information, which cannot be used for modelling, was dropped from the database. 

While cleaning the reviews data I realized, that unfortunately the user name of the review is not unique: it resulted of hundreds of Davids or Michaels. Due to this challenge, I couldn’t do a recommender system with collaborative filtering. I had to pivot my initial approach which led me to create a content-based recommender system instead which uses the course features and the new clusters of the courses. I utilize the reviews data to find the courses the users took, and recommend them other courses similar to their previous ones. 

### Step 3: EDA

For the numeric features, the distribution and the boxplot was used. For the categorical features a barchart was plotted.  The following graphs show an example of a numeric and a categorical feature:

**Summary of the most important findings:**
* there are around 900 courses with no reviews/ratings, but most of them are between rating 4 and 4.5
* The price ranges between 0 and 199 EUR
* The top 10 courses according to the number of subscribers are presented in the chart below:

![Top 10 courses](/images/EDA_barh_subscribers.png)

* Most courses don't have any quizzes or practice tests
* The average age of a course is 26 months (since it was published). There are more recently published courses than older ones.
* Courses are divided into 16 subcategories. The two most frequent categories are Finance and Entrepreneurship.
  * Two subcategories have an average price higher than 100$ : The subcategory Data & analytics with 112$, and Project Management with 104$
  * In terms of total potential earnings for tutors Data & Analytics and Entrepreneurship stand out from the crowd.
  * The total number of subscribers are the highest in the category of Entrepreneurship (1.) and in Data & Analytics (2.)
  * There is not much difference between the average ratings of the courses in each subcategory. The highest average ratings are in the subcategories Media and Communications

![Subcategories](/images/subcategories_barchart.png)

* These two attributes were analyzed deeper: “objectives” and “description”, since one of these features will be the basis of the clustering part in Step 4. The texts were tokenized and stemmed using the Natural Language Toolkit (nltk library) in Python. After removing the stopwords and punctuations, the following plots show the top 25 most popular words in the subcategories ‘Data & Analytics’ and ‘Finance’.  

![Wordcloud data](/images/wordcloud_data.png) 
![Wordcloud finance](/images/wordcloud_finance.png) 

**Most important findings on the reviews dataset:**
* Most users (more than 600.000) gave only one review, but there are couple of user names, who have plenty of reviews: the most common username is David with more than 400 reviews.
* Most courses have only a few reviews - 90% of the courses have less than 300 reviews.

### Step 4: Clustering:

I tried to analyse both the objectives and the descriptions of the courses to find new clusters. Finally the clustering algorithm based on the course descriptions showed better results.

The texts were first tokenized and stemmed, the stopwords and punctuations were removed. Afterwards the TfidfVectorizer was applied on the data, which helps to identify words which are frequent in the text but rare in the corpus. Based on this frequency-matrix I applied the k-means and hierarchical clustering algorithms. 

For the k-means algorithm I tried out multiple k-s (number of clusters). I checked how the inertia (within-cluster sum-of-squares) changes to look for an optimal number of clusters. According to the elbow method, the line is an arm and the "elbow" on the arm is the value of k that is the best. Finally I chose k=8  to be the optimal number of clusters.

![Inertia](/images/inertia.png) 

After fitting a k-means algorithm with k=8 clusters, the following graph shows the distribution of the clusters with their most common words in the description. The most courses can be found in cluster 0. The top 5 words in each cluster are in the label to help to identify what kind of courses belong the the clusters. 

![kmeans8](/images/bar_kmeans8_words.png) 

The connection between the clusters and the subcategories is plotted on the heatmap below. Courses in association with Data&Analytics are in cluster 3. In cluster 1 one branch of Finance is represented with popular words like trading, stock, forex and options.

![heatmap](/images/heatmap8.png)

To plot the clustering, I used a dimension reduction technique called PCA (Principal Component Analysis)  and kept the first 2 principle components. The two components (from the total 1000 components) explain more than 4% of the total information. I plotted the clusters on a sample data to check if the clusters can be well distinguished in this reduced dimension. As the next graph shows, almost all of the clusters are distributed nicely in 2-D.

![PCA](/images/pca.png)

### Step 5: Building a recommender system: 

There are two types of recommender systems: collaborative filtering and content/item based recommender systems.
* **Collaborative filtering** : uses the similarities in users’ behaviors and preferences to predict what users will like.
* **Content-based filtering:** use the description of the item and a profile of the user’s preferences to recommend new items

As I already mentioned, the reviews data (with the users’ behaviors) was not appropriate to build a recommender system with collaborative filtering. I used the clusters of step 4 and other features of the courses to build a content-based recommender system. 

Before building the recommender system, I transformed the feature matrix: 
1. Introduced dummy variables: 
I transformed the categorical features to dummies, that we can use these features as well in the recommender system. I also transformed the new clusters, since the cluster number doesn’t have any meaning (e.g. there is no relationship between cluster 1 and cluster 2)
2. Scaled the features:
Since the features of the courses have different magnitude, it is important to scale the feature matrix: the price of the courses varies between 0 and 199, but the average rating has a range between 0 and 5. I used the StandardScaler from the scikit-learn library, which standardizes all features by removing their mean and scaling them to unit variance.
3. I also defined a similarity measure to compare the courses:
I used the cosine similarity, which calculates the cosine of the angle between two vectors projected in a multidimensional space. In this context, the two vectors I am talking about are the arrays containing the transformed features of the courses.

There are 2 functions, which can be used to recommend courses:
* Function *recommend_for_user* recommends courses for the user based on his/her previous courses.
* Function *recommend_courses* recommends courses based on another course_id. This function takes the course_id instead of the user_name as input and looks for the courses that are similar to the original course.


## Refinement of the model

As further development the following improvements could be taken into consideration:
* Downloading courses from other categories to involve them in the clustering 
* Trying out other tokenizers and stemmers before applying the clustering algorithms
* Investigating the reviews dataset: identify unique user_names based on the taken courses










