import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('C:/Users/91888/Fake-Job-Posting-App/data/fake_job_postings.csv')
df.shape

df.head()

#Fraud and Real visualization
sns.countplot(df.fraudulent).set_title('Real & Fradulent')
fake=df['fraudulent'].sum()
real=len(df)-fake
print("real: {0}, fake: {1}".format(fake, real))
df['country'] = df.location.str.split(',').str[0]
country = dict(df.country.value_counts()[:11])
plt.figure(figsize=(8,6))
plt.title('Country-wise Job Posting', size=20)
plt.bar(country.keys(), country.values())
plt.ylabel('No. of jobs', size=10)
plt.xlabel('Countries', size=10)

#Visualize the required experiences in the jobs
experience = dict(df.required_experience.value_counts())
plt.figure(figsize=(10,5))
plt.bar(experience.keys(), experience.values())
plt.title('No. of Jobs with Experience')
plt.xlabel('Experience', size=10)
plt.ylabel('No. of jobs', size=10)
plt.xticks(rotation=35)
plt.show()

#Most frequent jobs
print("Most Frequent Jobs\n",df.title.value_counts()[:10])

#Titles and count of fraudulent jobs
print("Titles and count of fraudulent jobs\n",df[df.fraudulent==1].title.value_counts()[:10])

#Titles and count of real jobs
print("Titles and count of real jobs\n",df[df.fraudulent==0].title.value_counts()[:10])

#The dataset is very unbalanced. We will attempt to balance it by having equal of each real and fake postings.
df = df.sample(frac=1).reset_index(drop=True)
df = df.sample(frac=1).reset_index(drop=True)

df=df.set_index('job_id')

dropped = 0

for index, row in df.iterrows():
  if (df['fraudulent'][index] == 0) and (dropped < real-fake):
    df.drop(index, inplace=True)
    dropped+=1

fake=df['fraudulent'].sum()
real=len(df)-fake
print("real: {0}, fake: {1}".format(fake, real))

#Clean the text fields
import re

def text_tokenizer(text):
  if not (text == "" or pd.isnull(text)): 
    text = re.sub(r'URL_[A-Za-z0-9]+', ' ', text)
    return re.sub(r'[^A-Za-z0-9]+', ' ', text).lower().strip()

print(df.columns)

cols = ['title', 'location', 'department', 'company_profile', 'description', 'requirements', 'benefits', 'employment_type',
       'required_experience', 'required_education', 'industry', 'function']

for col in cols:
  for index, row in df.iterrows():
    cleaned = text_tokenizer(df[col][index])
    df[col] = df[col].replace(df[col][index],cleaned)

df = df.sample(frac=1).reset_index(drop=True)

df['text'] =  df['text'] = df['title'] + ' ' + df['location'] + ' ' + df['department'] + ' ' + df['company_profile'] + ' ' + df['description'] + ' ' + df['requirements'] + ' ' + df['benefits'] + ' ' + df['employment_type'] + ' ' + df['required_experience']+ ' ' + df['required_education'] + ' ' + df['industry'] + ' ' + df['function']

df = df.drop(['location','department','company_profile','description','requirements','benefits','employment_type', 'required_experience','required_education', 'industry', 'function'], 1)
df.to_csv('C:/Users/91888/Fake-Job-Posting-App/data/fake_job_postings_cleaned.csv')

#Splitting into test, train val
print("total num of rows: {0}, train size: {1}, test size: {2}".format(len(df), len(df)*0.8, len(df)*0.2))

train = df[:1385]
train = train.sample(frac=1).reset_index(drop=True)
test = df[1386:]
test = test.sample(frac=1).reset_index(drop=True)

from sklearn.feature_extraction.text import TfidfVectorizer

# Create feature vectors
vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)
train_vectors = vectorizer.fit_transform(train['text'].values.astype('U'))
test_vectors = vectorizer.transform(test['text'].values.astype('U'))

import time
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# Perform classification with SVM
classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
classifier_linear.fit(train_vectors, train['fraudulent'])
t1 = time.time()
prediction_linear = classifier_linear.predict(test_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1

# results
print("Classification Accuracy:", accuracy_score(test['fraudulent'],prediction_linear))
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
report = classification_report(test['fraudulent'], prediction_linear, output_dict=True)
print('positive: ', report['1'])
print('negative: ', report['0'])


import joblib
joblib.dump(classifier_linear, 'C:/Users/91888/Fake-Job-Posting-App/model.joblib')
joblib.dump(vectorizer, 'C:/Users/91888/Fake-Job-Posting-App/vectorizer.joblib')



