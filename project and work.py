from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import ast
import nltk
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


def stem(text):
    y=[]
    
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

ted = pd.read_csv('ted_main.csv')

pd.set_option('display.max_columns', 7)
print(ted.head())
print(ted.shape)
print(ted.head(1))

print(ted.info())


print("Number duplicate entry:",ted.duplicated().sum())

print("\n description :",ted.iloc[0].description)
print("\n ratings :",ted.iloc[0].ratings)
print("\n tags :",ted.iloc[0].tags)
print("\n event :",ted.iloc[0].event)


#preprocessing in ratings column
def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

ted['ratings'] = ted['ratings'].apply(convert)

print(ted['ratings']) 
print(ted['tags'])
print(ted['event'])
print(ted['description'])

 

ted['description']=ted['description'].apply(lambda x:x.split())
ted['tags'] = ted['tags'].apply(lambda x:x.split())
ted['event'] = ted['event'].apply(lambda x:x.split())
print(ted['description'])
print(ted.info())


ted['detail']=ted['description']+ted['tags']+ted['event']+ted['ratings']
print(ted['detail'])
new_df = ted[['name','main_speaker','detail','url']]

print(new_df.head())
print(new_df.head().shape)
print(new_df.info())

print(new_df['detail'])
new_df['detail']=new_df['detail'].apply(lambda x:" ".join(x))
print(new_df['detail'][0])
new_df['detail']=new_df['detail'].apply(lambda x:x.lower())
print(new_df['detail'][0])

cv= CountVectorizer(max_features=5000,stop_words='english')

vectors = cv.fit_transform(new_df['detail']).toarray()
print("Vectors is :",vectors)

print(vectors[0])
print(cv.get_feature_names_out())
print(new_df['detail'][0])
new_df['detail']=new_df['detail'].apply(stem)
print(new_df['detail'][0])
print(cv.get_feature_names_out())



#calculating the cosine distance
similarity = cosine_similarity(vectors)

print(similarity[1])

def recomend(video):
    video_index=new_df[new_df['name'] == video].index[0]
    distances = similarity[video_index]
    video_list= sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in video_list:
        print(new_df.iloc[i[0]])

recomend('Bobby McFerrin: Watch me play ... the audience!')
recomend('Tim Birkhead: The early birdwatchers')
