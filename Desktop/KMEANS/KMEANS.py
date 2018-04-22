# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 02:33:35 2018

@author: Siddharth
"""

import numpy as np
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

from sklearn.preprocessing import LabelEncoder, MinMaxScaler


x = pd.read_excel("C:/Users/Siddharth/Desktop/FYI_PROJECT/data_presentation.xlsx")
y = pd.read_excel("C:/Users/Siddharth/Desktop/FYI_PROJECT/bin_category_data.xlsx")

x = pd.merge(x,y,how = 'left',left_on='DISPLAY NAME',right_on='DISPLAY NAME')


#replacing null values from actual views and factors to zero
replace_cols = list(x.loc[:,'Actual FB Views':'Military_y'])
x[replace_cols] = x[replace_cols].fillna(0)

#replacing empty spaces to NULL
x = x.fillna('NULL')


#dropping the redundant influencer categories
#drop_cols = list(x.loc[:,'Automotive':'NULL_value'])
#x = x.drop(drop_cols,axis = 1)


#replace select to NULL for categorical variables
replace_select = list(x.loc[:,'GENDER':'Political Affiliation'])
x[replace_select] = x[replace_select].replace('Select','NULL')

#drop log columns
x=x.drop('LOG',axis = 1)
x=x.drop('LOG FB',axis = 1)
x=x.drop('LOG TWITTER',axis = 1)
x=x.drop('LOG INSTAGRAM',axis = 1)
x=x.drop('LOG BLOG',axis = 1)
x=x.drop('LOG GOOGLE',axis = 1)
x=x.drop('LOG YOUTUBE',axis = 1)




#standardize the numerical variables
le = LabelEncoder()
scaler = MinMaxScaler()
std_cols = list(x.loc[:,'AVN':'Youtube_Followers']) + list(x.loc[:,'Actual FB Views':'Actual Youtube Views'])
x[std_cols] = scaler.fit_transform(x[std_cols])


#Label encoding for categorical variables
#label_enc = list(x.loc[:,'How long have you been blogging?':'Political Affiliation'])
#x[label_enc] = x[label_enc].astype(str)
#
#x['EN-GENDER'] = le.fit_transform(x['GENDER'])
#x['EN-BLOG'] = le.fit_transform(x['How long have you been blogging?'])
#x['EN-EDUCATION'] = le.fit_transform(x['Highest Level Of Education Completed'])
#x['EN-EMPLOYEMENT'] = le.fit_transform(x['Employment Status'])
#x['EN-HHI'] = le.fit_transform(x['HHI'])
#x['EN-RACE'] = le.fit_transform(x['Race'])
#x['EN-PA'] = le.fit_transform(x['Political Affiliation'])

#l_enc = list(x.loc[:,'EN-GENDER':'EN-PA'])
#x[l_enc] = scaler.fit_transform(x[l_enc])


#x = x.drop(label_enc,axis = 1)

#clustering criteria
cols = list(x.loc[:,'AVN':'Youtube_Followers']) + list(x.loc[:,'Actual FB Views':'Military_y']) 
x_subset = x[cols]
#
#
#
##optimal no of clusters for k-means : Elbow Method
cluster_range = range(5,100)
cluster_errors = []
for num_clusters in cluster_range :
    clusters = KMeans(num_clusters, random_state = 0).fit(x_subset)
    cluster_errors.append( clusters.inertia_ )

clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )

plt.figure(figsize=(12,6))
plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )




##using k =12
#kmeans = KMeans(n_clusters = 12, random_state = 0).fit(x_subset)
#centers = kmeans.cluster_centers_
#labels = kmeans.labels_
#
#def distance_to_centroid(row, centroid):
#    row = row[['AVN',
#                'FB_Followers',
#                'Twitter_Followers',
#                'Instagram_Followers',
#                'Blog_Followers',
#                'GooglePlus_Followers',
#                'Youtube_Followers',
#                'Actual FB Views',
#                'Actual Twitter Views',
#                'Actual Insta Views',
#                'Actual Blog Views',
#                'Actual Google Views',
#                'Actual Youtube Views',
#                'Factor 1',
#                'Factor2',
#                'Factor 3',
#                'Factor4',
#                'Factor5',
#                'Factor 6',
#                'Military_y']]
#    return euclidean(row, centroid)
#
#x['distance_to_center0'] = x.apply(lambda r: distance_to_centroid(r,
#    centers[0]),1)
#
#x['distance_to_center1'] = x.apply(lambda r: distance_to_centroid(r,
#    centers[1]),1)
#x['distance_to_center2'] = x.apply(lambda r: distance_to_centroid(r,
#    centers[2]),1)
#
#x['distance_to_center3'] = x.apply(lambda r: distance_to_centroid(r,
#    centers[3]),1)
#x['distance_to_center4'] = x.apply(lambda r: distance_to_centroid(r,
#    centers[4]),1)
#
#x['distance_to_center5'] = x.apply(lambda r: distance_to_centroid(r,
#    centers[5]),1)
#
#x['distance_to_center6'] = x.apply(lambda r: distance_to_centroid(r,
#    centers[6]),1)
#
#x['distance_to_center7'] = x.apply(lambda r: distance_to_centroid(r,
#    centers[7]),1)
#x['distance_to_center8'] = x.apply(lambda r: distance_to_centroid(r,
#    centers[8]),1)
#
#x['distance_to_center9'] = x.apply(lambda r: distance_to_centroid(r,
#    centers[9]),1)
#x['distance_to_center10'] = x.apply(lambda r: distance_to_centroid(r,
#    centers[10]),1)
#
#x['distance_to_center11'] = x.apply(lambda r: distance_to_centroid(r,
#    centers[11]),1)
#
#
#
#
#
#x["Eucledian Distance"] = x.loc[:,'distance_to_center0':'distance_to_center11'].min(axis = 1)
#
#
#
#remove_dist = list(x.loc[:,'distance_to_center0':'distance_to_center11'])
#x = x.drop(remove_dist,axis = 1)
#
#x['label'] = kmeans.labels_
#x['label'] = x['label'].astype(str)
#
#
##x = x.sort_values(['label','Eucledian Distance'], ascending = [1,1])
##print(x.head(10))
#
## sum of influencer categories per label 
#replace_select = list(x.loc[:,'Automotive':'NULL_value'])
#y = x.loc[:,'Automotive':'NULL_value'].groupby(x['label']).agg('sum')
#
#
#
#
#writer = ExcelWriter('influencer count per label.xlsx')
#y.to_excel(writer,'Sheet1',index=False)
#writer.save()


##
##
##
##
#writer = ExcelWriter('category_performance_final.xlsx')
#x.to_excel(writer,'Sheet1',index=False)
#writer.save()

