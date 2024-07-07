import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.linear_model import ElasticNet
from sklearn.metrics import roc_auc_score
import time
from datetime import timedelta

num_folds = 5
start_time = time.time()

for i in range(num_folds):
    process_time = time.time()
    print(f"Split {i+1} processing...")
    file_path = f'split_{i+1}/train.tsv'
    train = pd.read_csv(file_path, sep='\t', header=0, dtype=str)
    train['review'] = train['review'].str.replace('&lt;.*?&gt;', ' ', regex=True)

    file_path = f'split_{i+1}/test.tsv'
    test = pd.read_csv(file_path, sep='\t', header=0, dtype=str)
    test['review'] = test['review'].str.replace('&lt;.*?&gt;', ' ', regex=True)

    file_path = f'split_{i+1}/test_y.tsv'
    test_y = pd.read_csv(file_path, sep='\t', header=0, dtype=str)


    file_path = 'myvocab.txt'
    with open(file_path, 'r') as file:  
        myvocab = [line.strip() for line in file]

    stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "their", "they", "his", "her", "she", "he", "a", "an", "and", "is", "was", "are", "were", "him", "himself", "has", "have", "it", "its", "the", "us"]

    vectorizer = CountVectorizer(
        preprocessor=lambda x: x.lower(),
        stop_words=stop_words,
        ngram_range=(1, 4),
        token_pattern=r"\b[\w+\|']+\b"            
    )
    vectorizer.fit(myvocab)
    dtm_train = vectorizer.transform(train['review'])

    
    #ridge = Ridge(alpha=0.1)
    #ridge.fit(dtm_train, train['sentiment'])

    dtm_test_trimmed = vectorizer.transform(test['review'])

    logistic_reg = LogisticRegression(penalty='l2', C=0.1, max_iter=1000)  # You can adjust the regularization strength (C)
    logistic_reg.fit(dtm_train, train['sentiment'])
    predictions_logistic = logistic_reg.predict_proba(dtm_test_trimmed)[:, 1]

    auc_score_log = roc_auc_score(test_y['sentiment'], predictions_logistic)

    print(f"AUC Score Logistic: {auc_score_log}")

    print(f"Time Taken: {str(timedelta(seconds=time.time() - process_time))}")
