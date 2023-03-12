import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

True_news = pd.read_csv('True.csv')
Fake_news = pd.read_csv('Fake.csv')

True_news['label'] = 0
Fake_news['label'] = 1

dataset1 = True_news[['text','label']]
dataset2 = Fake_news[['text','label']]

dataset = pd.concat([dataset1 , dataset2])

dataset.isnull().sum()

dataset['label'].value_counts()

# shuffle / resample
dataset = dataset.sample(frac = 1)

ps = WordNetLemmatizer()

nltk.download('stopwords')
nltk.download('omw-1.4')

stopwords = stopwords.words('english')

nltk.download('wordnet')

def cleaning_data(row):
    
    row = row.lower()
    row = re.sub('[^a-zA-Z]', ' ', row)
    token = row.split()
    
    # Lemmatize the word and remove stop words like a, an, the, is, are
    news = [ps.lemmatize(word) for word in token if not word in stopwords]
    cleanned_news = ' '.join(news)
    
    return cleanned_news


dataset['text'] = dataset['text'].apply(lambda x: cleaning_data(x))


vectorizer = TfidfVectorizer(max_features=50000, lowercase=False, ngram_range = (1, 2))


X = dataset.iloc[:10000,0] #35000
y = dataset.iloc[:10000,1] #35000

# print(X.head())
# print(y.head())

train_data, test_data, train_label, test_label = train_test_split(X, y, test_size=0.2, random_state=0)

vec_train_data = vectorizer.fit_transform(train_data)
vec_train_data = vec_train_data.toarray()

vec_test_data = vectorizer.transform(test_data).toarray()

training_data = pd.DataFrame(vec_train_data , columns=vectorizer.get_feature_names_out())
testing_data = pd.DataFrame(vec_test_data , columns= vectorizer.get_feature_names_out())


# the model
clf = MultinomialNB()

clf.fit(training_data, train_label)

y_pred = clf.predict(testing_data)

#MultinomialNB

pd.Series(y_pred).value_counts()
test_label.value_counts()

print(classification_report(test_label, y_pred))

y_pred_train = clf.predict(training_data)

print(classification_report(train_label, y_pred_train))

accuracy_score(train_label, y_pred_train)
accuracy_score(test_label, y_pred)


news = cleaning_data(str("Joe Biden Appoints 2 Indian-American CEOs to US ADvisoru Committee."))


single_prediction = clf.predict(vectorizer.transform([news]).toarray())
print(single_prediction)


# export the clf model
joblib.dump(clf, 'clf_model.plk')
model = joblib.load('clf_model.plk')





