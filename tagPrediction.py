# %% imports
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
#nltk.download()    # download packages if needed

# %% read training set
train = pd.read_csv("train.tsv", sep='\t').fillna(" ")
trainCount = train['description'].count()

# %% pre-process description
description = train['description']

def description_to_words(raw_description):
    # remove non-letters
    letters_only = re.sub("[^a-zA-Z0-9]", " ", raw_description) 
    # convert to lower case and split into words
    words = letters_only.lower().split()
    # remove stop words
    stops = set(stopwords.words("english"))  
    meaningful_words = [w for w in words if not w in stops] 
    # stem words
    stemmer = SnowballStemmer('english')
    stem_words = [stemmer.stem(word) for word in meaningful_words]
    return( " ".join(stem_words)) 
    
description_text = description.apply(lambda x: description_to_words(x))

# %% categorize tags, each mutally exclusive tags belongs to a group
tags = train[['tags']]
timeType = ['part-time-job', 'full-time-job']
payType = ['hourly-wage', 'salary']
eduType = ['associate-needed', 'bs-degree-needed', 'ms-or-phd-needed', 'licence-needed']
expType = ['1-year-experience-needed', '2-4-years-experience-needed', '5-plus-years-experience-needed']
supType = ['supervising-job']

def getType(tagList, jobType):
    result = [val for val in tagList.split(" ") if val in jobType]
    if len(result) == 0:
        result.append(" ")
    return result[0]

tags['timeType'] = tags.apply(lambda x: getType(x['tags'], timeType), axis=1)
tags['payType'] = tags.apply(lambda x: getType(x['tags'], payType), axis=1)
tags['eduType'] = tags.apply(lambda x: getType(x['tags'], eduType), axis=1)
tags['expType'] = tags.apply(lambda x: getType(x['tags'], expType), axis=1)
tags['supType'] = tags.apply(lambda x: getType(x['tags'], supType), axis=1)

# %% transform words to numbers
vectorizer = CountVectorizer(analyzer = 'word',ngram_range=(2,4), max_features=5000)
vectorizer.fit(description_text)
vectorized_training_data = vectorizer.transform(description_text)

# %% run random forest on each type
num_estimator = 95
timeTypeForest = RandomForestClassifier(n_estimators=num_estimator)
timeTypeForest = timeTypeForest.fit(vectorized_training_data, tags['timeType'])

payTypeForest = RandomForestClassifier(n_estimators=num_estimator)
payTypeForest = payTypeForest.fit(vectorized_training_data, tags['payType'])

eduTypeForest = RandomForestClassifier(n_estimators=num_estimator)
eduTypeForest = eduTypeForest.fit(vectorized_training_data, tags['eduType'])

expTypeForest = RandomForestClassifier(n_estimators=num_estimator)
expTypeForest = expTypeForest.fit(vectorized_training_data, tags['expType'])

supTypeForest = RandomForestClassifier(n_estimators=num_estimator)
supTypeForest = supTypeForest.fit(vectorized_training_data, tags['supType'])


# %% work on test data
testData = pd.read_csv("test.tsv", sep='\t').fillna(" ")
testCount = testData.count()

# pre-process test data
testData_text = testData['description'].apply(lambda x: description_to_words(x))
vectorized_test_data = vectorizer.transform(testData_text)

# %% use random forest to make predictions
timeResult = timeTypeForest.predict(vectorized_test_data)
payResult = payTypeForest.predict(vectorized_test_data)
eduResult = eduTypeForest.predict(vectorized_test_data)
expResult = expTypeForest.predict(vectorized_test_data)
supResult = supTypeForest.predict(vectorized_test_data)

# %% output result to csv
dfResult = pd.DataFrame({'time': timeResult, 'pay': payResult, 'edu': eduResult, 'exp': expResult, 'sup': supResult})
# convert results to a single column
dfResult['tags'] = (dfResult['time'] + " " + dfResult['pay'] + " " + dfResult['edu'] + " " + dfResult['exp'] + " " + dfResult['sup'])
dfResult['tags'] = dfResult['tags'].apply(lambda x: " ".join(x.split()))
dfResult['tags'].to_csv("result.tsv", index=False, header='tags', sep='\t')