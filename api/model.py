# Import dependencies
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

eng_stopwords = set(stopwords.words("english"))
lem = WordNetLemmatizer()

def clean_comment(text):
    
    #split into words
    tokens = word_tokenize(text)
    #change words to lower case and lemmatize
    stemmed = [lem.lemmatize(word.lower()) for word in tokens]
    #remove stop words
    words = [w for w in stemmed if not w in eng_stopwords]
    #remove anything non-alphabetic
    clean_words = [word for word in words if word.isalpha()]
    #append to string
    clean_comment = " ".join(clean_words)
    
    return clean_comment


#import the data as a DataFramae with 'id' as the index
train_df = pd.read_csv('../data/train.csv', index_col='id')
#fill in any null values
train_df["comment_text"].fillna("unknown", inplace=True)

#add up the tags for each comment.
rowsums = train_df.iloc[:,1:].sum(axis=1)
#if the comment has no tags then it is non-toxic
train_df['non_toxic'] = (rowsums==0) 
#convert the boolean column to 1 or 0
train_df = train_df.applymap(lambda x: 1 if x == True else x)
train_df = train_df.applymap(lambda x: 0 if x == False else x)

#remove any comment with multiple tags
train_df = train_df[train_df.iloc[:,1:].sum(axis=1) == 1]
#Map the tag into a unique integer
train_df['severe_toxic'] = train_df['severe_toxic'].apply(lambda x: 2 if x == 1 else x)
train_df['obscene'] = train_df['obscene'].apply(lambda x: 3 if x == 1 else x)
train_df['threat'] = train_df['threat'].apply(lambda x: 4 if x == 1 else x)
train_df['insult'] = train_df['insult'].apply(lambda x: 5 if x == 1 else x)
train_df['identity_hate'] = train_df['identity_hate'].apply(lambda x: 6 if x == 1 else x)
train_df['non_toxic'] = train_df['non_toxic'].apply(lambda x: 7 if x == 1 else x)
#create one column with the classification
train_df['classification'] = train_df['toxic']+train_df['severe_toxic']+train_df['obscene']+train_df['threat']+train_df['insult']+train_df['identity_hate']+train_df['non_toxic']

#clean the comments
train_df.comment_text = train_df.comment_text.apply(lambda x: clean_comment(x))

#The model
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

nb_classifier = MultinomialNB()
count_vectorizer = CountVectorizer(stop_words='english')

y = train_df['classification']
count_train = count_vectorizer.fit_transform(train_df['comment_text'])

# Fit the classifier to the training data
nb_classifier.fit(count_train, y)

# Save your model
from sklearn.externals import joblib
joblib.dump(nb_classifier, 'toxic_model.pkl')
print("Model dumped!")
