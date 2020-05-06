import pandas as pd
import matplotlib.pyplot as plt

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
df['category_id'] = df['cuisine'].factorize()[0]

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

features = tfidf.fit_transform(df.ingredients)
labels = df.category_id
print(features.shape)

from sklearn.feature_selection import chi2
import numpy as np

#Return the top 3 unigrams and bigrams associated with a cuisine
N = 3
for cuisine, category_id in sorted(category_to_id.items()):
    features_chi2 = chi2(features, labels == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("# '{}':".format(cuisine))
    print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
    print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))


#Before checking the test set, split the training set to see if program actually works
#Try to do a prediction using Multinomial Naive Bayes method
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(df['ingredients'], df['cuisine'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)

#Compare the MultinomialNB accuracy with Linear SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score


models = [
    LinearSVC(),
    MultinomialNB(),
]


CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
        cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
import seaborn as sns
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df,
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()

cv_df.groupby('model_name').accuracy.mean()

#Linear SVC performed before, thats what we'll use from now on
from sklearn.model_selection import train_test_split

model = LinearSVC()

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#Display confusion matrix in a heatmap
#Cool visualization
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.cuisine.values, yticklabels=category_id_df.cuisine.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

"""
#optional to run this
#Will print all the misclassifications
from IPython.display import display
for predicted in category_id_df.category_id:
    for actual in category_id_df.category_id:
        if predicted != actual and conf_mat[actual, predicted] >= 10:
            print("'{}' predicted as '{}' : {} examples.".format(id_to_category[actual], id_to_category[predicted], conf_mat[actual, predicted]))
            display(df.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['cuisine', 'ingredient']])
            print('')
"""

#We perform a Chi-2 test again to see any difference
model.fit(features, labels)
N = 3
for cuisine, category_id in sorted(category_to_id.items()):
    indices = np.argsort(model.coef_[category_id])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
    bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
    print("# '{}':".format(cuisine))
    print("  . Top unigrams:\n       . {}".format('\n       . '.join(unigrams)))
    print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))


#Cool table showing the accuracy per cuisine and general accuracy
from sklearn import metrics
print(metrics.classification_report(y_test, y_pred,
                                    target_names=df['cuisine'].unique()))

print(model.predict(test_features))