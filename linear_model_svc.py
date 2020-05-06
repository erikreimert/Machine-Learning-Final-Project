import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC

#Work with training set manipulation
df = pd.read_csv(r"data_file_nocomma.csv", encoding = 'latin-1')
df.head()

#Analyze non-null values
df = df[pd.notnull(df['ingredients'])]
df.info()

#We won't need the ID to train the data set
col = ['cuisine', 'ingredients']
df = df[col]

#Transform all cuisines into a 1d array with labels
df.columns = ['cuisine', 'ingredients']
df['category_id'], cuisine_idx = df['cuisine'].factorize()

#Testing set manipulation
test_file = pd.read_csv(r"test_file.csv", encoding = 'latin-1')
test_file.head()

#save Id to hstack later
test_id = test_file['id']

#Drop duplicate categories
#Turn into dictionaries
from io import StringIO
category_id_df = df[['cuisine', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'cuisine']].values)

#Transform the daa\ta to be used in Chi-2 test
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

all_ings = pd.concat([test_file.ingredients, df.ingredients])
all_features = tfidf.fit_transform(all_ings)
train_features = tfidf.transform(df.ingredients)
train_labels = df.category_id
test_features = tfidf.transform(test_file.ingredients)

model = LinearSVC()
model.fit(train_features, train_labels)

preds = cuisine_idx[model.predict(test_features)]
print(preds)

output = pd.DataFrame(preds, index=test_id)
print(output)

output.to_csv("linear_pred.csv")