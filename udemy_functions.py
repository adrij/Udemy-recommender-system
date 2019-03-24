import requests
import ast
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from operator import itemgetter
from collections import Counter
import matplotlib
import squarify
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import StandardScaler


#get the data from the Udemy 
def get_data(url, username, pw):
    r = requests.get(url, auth=(username,pw))
    data_json=r.json()
    return data_json

#transform dataframe columns with dict/list values 
def transform_col(col, col_key=None):
	if col_key:
		return col.apply(lambda x: ast.literal_eval(x).get(col_key))
	else:
		return col.apply(lambda x: ast.literal_eval(x))

def get_float(text):
    r=re.search('\d+\.*\d*', text)
    if r:
        return float(r.group(0))
    else:
        return np.nan

def remove_tags(text):
    tag_re = re.compile(r'<[^>]+>')
    if text==text:
        return tag_re.sub('', text).replace('\n',' ').replace('\xa0',' ').replace('\t',' ')

#functions for stemming
def tokenize(text):
    stemmer=SnowballStemmer('english')
    return [stemmer.stem(word) for word in word_tokenize(text.lower())]

def tokenize_only(text):
    return [word for word in word_tokenize(text.lower())]

def combine_list(l):
    new_str=""
    for item in l:
        new_str=new_str+' '+str(item) 
    return new_str

def vocab_stem(text):
    stemmer=SnowballStemmer('english')
    total_stemmed = []
    total_tokenized = []
    for i in text:
        obj_stemmed = tokenize(i) 
        total_stemmed.extend(obj_stemmed) 
        obj_tokenized = tokenize_only(i)
        total_tokenized.extend(obj_tokenized)
    vocab_frame = pd.DataFrame({'words': total_tokenized}, index = total_stemmed)
    #print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
    return vocab_frame 

def drop_words(vocab_frame):
    vocab_frame=vocab_frame.reset_index()
    vocab_frame.columns = ['index','words']
    vocab_frame=vocab_frame.drop_duplicates(subset='index', keep='first').set_index('index')
    #print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
    return vocab_frame

def get_words_count(text, StopWords, vocab_frame):
    list_words=[tokenize(items) for items in text]

    word=[word for words in list_words for word in words if word not in StopWords]
    words_dict=Counter(word)
    if vocab_frame is None:
        return sorted(words_dict.items(), key=itemgetter(1), reverse=True)
    else:
        words_dict_new={}
        for k,v in words_dict.items():
            word_new=vocab_frame.loc[k].values.tolist()[0]
            words_dict_new[word_new]=v
        return sorted(words_dict_new.items(), key=itemgetter(1), reverse=True)

def top_words_graph(df_courses, attribute, comb_list, kind, StopWords, vocab_frame):
    for subcat in df_courses['primary_subcategory'].unique():
        temp=df_courses[df_courses['primary_subcategory']==subcat]
        if comb_list:
            text=temp[attribute].apply(combine_list).values
        else:
            text=temp[attribute].values
        top_words=get_words_count(text, StopWords, vocab_frame)[:25]
        plt.subplots(figsize=(10,8))
        if kind=='bar':
            plt.barh(range(len(top_words)), [val[1] for val in top_words], align='center')
            plt.yticks(range(len(top_words)), [val[0] for val in top_words])
            plt.xlabel("Number of occurences")    
        else:
            top_words=dict(top_words)
            wordcloud = WordCloud(width=900,height=500, margin=0).generate_from_frequencies(top_words)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
        plt.title(subcat)

def get_label(index, vocab_frame, word_features):
    return vocab_frame.loc[word_features[index]].values.tolist()[0]


def get_common_words(model, count_words):
    count_words_new=count_words*(-1)-1
    common_words = model.cluster_centers_.argsort()[:,-1:count_words_new:-1]
    return common_words

def print_common_words(common_words, word_features, vocab_frame, print_list=True):
    dict_cluster={}
    for num, centroid in enumerate(common_words):
        dict_cluster[num]=[get_label(word, vocab_frame, word_features) for word in centroid]
        if print_list:
            print(str(num) + ' : ' + ', '.join(dict_cluster[num]))
    if print_list==False:
        return dict_cluster
    
def plot_common_words(model, n_words, word_features, vocab_frame, df_courses, cluster_name):
    common_words=get_common_words(model, n_words)
    dict_cluster=print_common_words(common_words, word_features, vocab_frame, False)
    fig, ax=plt.subplots(figsize=(12,5))
    keys=df_courses[cluster_name].value_counts().sort_index().index
    values=df_courses[cluster_name].value_counts().sort_index().values
    colors=['b', 'g', 'y','r', 'k', 'grey', 'purple','orange', 'pink', 'brown']
    for j in range(len(keys)):
        ax.bar(keys[j], values[j], width=0.8, bottom=0.0, align='center', color=colors[j], alpha=0.4, label=dict_cluster[j]) 
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(['cluster '+str(k) for k in keys])
    ax.set_ylabel('Number of courses')
    ax.set_title('Distribution of clusters with the top ' + str(n_words) + ' words')
    plt.legend(fontsize=13)
    
def squarify_words(common_words, word_features, vocab_frame):
    colormaps=['Purples', 'Blues', 'Greens', 'Oranges', 'Reds','Greys', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu',
           'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    for num, centroid in enumerate(common_words):
        sizes=np.arange(10,10+len(centroid))
        cmap_name=colormaps[num]
        cmap = plt.get_cmap(cmap_name)
        labels=[get_label(word, vocab_frame, word_features) for word in centroid]
        mini=min(sizes)
        maxi=max(sizes)
        norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)
        colors = [cmap(norm(value)) for value in sizes]
        squarify.plot(sizes=sizes, label=labels,alpha=0.6, color=colors)
        plt.title("Most frequent words in cluster "+str(num))
        plt.show()

def heatmap_categories_cluster(cluster_name, df_courses, cmap ):
    clusters = df_courses.groupby([cluster_name, 'primary_subcategory']).size()
    fig, ax = plt.subplots(figsize = (30, 15))
    sns.heatmap(clusters.unstack(level = 'primary_subcategory'), ax = ax, cmap = cmap)
    ax.set_xlabel('primary_subcategory', fontdict = {'weight': 'bold', 'size': 24})
    ax.set_ylabel(cluster_name, fontdict = {'weight': 'bold', 'size': 24})
    for label in ax.get_xticklabels():
        label.set_size(16)
        label.set_weight("bold")
    for label in ax.get_yticklabels():
        label.set_size(16)
        label.set_weight("bold")   

def get_inertia(data, nClusterRange):
    inertias = np.zeros(len(nClusterRange))
    for i in range(len(nClusterRange)):
        model = KMeans(n_clusters=i+1, init='k-means++', random_state=1234).fit(data)
        inertias[i] = model.inertia_
    return inertias

def plot_inertia(kRange, inertia_Kmean):
    plt.figure(figsize=(10,8))
    plt.plot(kRange, inertia_Kmean, 'o-', color='seagreen', linewidth=3)
    #plt.plot([6], [testKmean[5]], 'o--', color='dimgray', linewidth=3)
    #plt.plot([1,6,11], [8520, 8170,7820], '--', color='k', linewidth=1)
    #plt.annotate("Let's try k=6", xy=(6, testKmean[5]), xytext=(6,7700),
             #size=14, weight='bold', color='dimgray',
             #arrowprops=dict(facecolor='dimgray', shrink=0.05))
    plt.xlabel('k [# of clusters]', size=18)
    plt.ylabel('Inertia', size=14)
    plt.title('Inertia vs KMean Parameter', size=14)

def print_titles_cluster(n_title, df_courses, cluster_name):
    for i in df_courses[cluster_name].unique():
        temp=df_courses[df_courses[cluster_name]==i]
        print(temp['published_title'].values[:n_title])

#functions for hierarchical clustering:
def get_linkage(X ):
    dist=pdist(X.todense(), metric='euclidean')
    z = linkage(dist, 'ward')
    return z

def plot_dendrogram(z, last_p_show, line_dist=None):
    # lastp is telling the algorithm to truncate using the number of clusters we set
    plt.figure(figsize=(20,10))
    plt.title('Dendrogram for attribute objectives')
    plt.xlabel('Data Index')
    plt.ylabel('Distance (ward)')
    dendrogram(z, orientation='top', leaf_rotation=90, p=last_p_show, truncate_mode='lastp', show_contracted=True);
    if line_dist!=None:
        plt.axhline(line_dist, color='k')

def plot_with_pca (X, labels, plot_n_sample):
    pca=PCA(n_components=2)
    X_2d=pca.fit_transform(X.todense())
    print('The explained variance through the first 2 principal comonent is {}.'
          . format(round(pca.explained_variance_ratio_.sum(),4)))
    df = pd.DataFrame(dict(x=X_2d[:,0], y=X_2d[:,1], label=labels)) 
    df_sample=df.sample(plot_n_sample)
    groups = df_sample.groupby('label')
    cluster_colors=['b', 'g', 'y','r', 'k', 'grey', 'purple','orange', 'pink', 'brown']
    fig, ax = plt.subplots(figsize=(17, 9)) 
    for name in np.arange(len(df_sample['label'].unique())):
        temp=df_sample[df_sample['label']==name]
        ax.plot(temp.x, temp.y, marker='o', linestyle='', ms=12, 
            label='cluster '+str(name), 
            color=cluster_colors[name], 
            mec='none', alpha=0.6)
        ax.set_aspect('auto')
        ax.tick_params(axis= 'x', which='both', bottom='off', top='off', labelbottom='off')
        ax.tick_params(axis= 'y', which='both', bottom='off', top='off', labelbottom='off')
    ax.legend(numpoints=1) 
    plt.title('Courses with PCA decompostion')

#functions for the recommender system
def normalize_features(df):
    df_norm = df.copy()
    for col in df_norm.columns:
        df_norm[col] = StandardScaler().fit_transform(df_norm[col].values.reshape(-1, 1))
    return df_norm

def recommend_courses(course_id, n_courses, df_courses, df_norm):
    n_courses=n_courses+1
    id_=df_courses[df_courses['id']==course_id].index.values
    title=df_courses[df_courses['id']==course_id]['published_title']
    X = df_norm.values
    Y = df_norm.loc[id_].values.reshape(1, -1)
    cos_sim = cosine_similarity(X, Y)
    df_sorted=df_courses.copy()
    df_sorted['cosine_similarity'] = cos_sim
    df_sorted=df_sorted.sort_values('cosine_similarity', ascending=False).reset_index(drop=True)

    return title, df_sorted.iloc[1:n_courses][['published_title', 'cosine_similarity']]

def recommend_for_user(user_name, n_courses, df_reviews, df_courses, df_norm):
    list_courses=df_reviews[df_reviews['user_name']==user_name]['course_id'].values
    len_courses=len(list_courses)
    index_courses=df_courses[df_courses['id'].isin(list_courses)].index
    for course_id in list_courses:
        title, df_recommend= recommend_courses(course_id, n_courses, df_courses, df_norm)
        print('The following courses are recommended after taking the course {} with the id {}:'
          .format(title.values[0],course_id))
        print(df_recommend)
        print()
    if len_courses>1:
        n_courses=n_courses+1
        df_temp=df_courses.copy()
        for i, course_id in enumerate(list_courses):
            id_=df_courses[df_courses['id']==course_id].index.values
            X = df_norm.values
            Y = df_norm.loc[id_].values.reshape(1, -1)
            cos_sim = cosine_similarity(X, Y)
            df_temp[i] = cos_sim
        temp_avg=df_temp.iloc[:,-len_courses:].mean(axis=1).values
        df_temp['avg_cos_sim']=temp_avg
        df_temp.drop(index=index_courses, inplace=True)
        df_temp=df_temp.sort_values('avg_cos_sim', ascending=False).reset_index(drop=True)
        print('The following courses are recommended after all taken courses:')
        print(df_temp.iloc[1:n_courses][['published_title', 'avg_cos_sim']])
