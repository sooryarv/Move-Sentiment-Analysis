import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

num_folds = 1
all_reviews = pd.DataFrame()

for i in range(num_folds):
    #print(f"Split {i+1} processing...")
    file_path = f'split_{i+1}/train.tsv'
    train = pd.read_csv(file_path, sep='\t', header=0, dtype=str)
    train['review'] = train['review'].str.replace('&lt;.*?&gt;', ' ', regex=True)

    file_path = f'split_{i+1}/test.tsv'
    test = pd.read_csv(file_path, sep='\t', header=0, dtype=str)
    test['review'] = test['review'].str.replace('&lt;.*?&gt;', ' ', regex=True)
    
    file_path = f'split_{i+1}/test_y.tsv'
    test_y = pd.read_csv(file_path, sep='\t', header=0, dtype=str)

    merged_test = pd.merge(test, test_y, on='id', how='inner')

    all_reviews = pd.concat([all_reviews, train])
    all_reviews = pd.concat([all_reviews, merged_test])

stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "their", "they", "his", "her", "she", "he", "a", "an", "and", "is", "was", "are", "were", "him", "himself", "has", "have", "it", "its", "the", "us"]

vectorizer = CountVectorizer(
    preprocessor=lambda x: x.lower(),
    stop_words=stop_words,
    ngram_range=(1, 4),
    min_df=0.001,
    max_df=0.5,
    token_pattern=r"\b[\w+\|']+\b"
)

dtm_train = vectorizer.fit_transform(all_reviews['review'])

lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.1)
lasso.fit(dtm_train, all_reviews['sentiment'])

# Get feature names and corresponding coefficients
feature_names = vectorizer.get_feature_names_out()
coefficients = lasso.coef_[0]

# Create a DataFrame to hold feature names and coefficients
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})


# Sort the DataFrame based on the absolute magnitude of coefficients
coef_df['Absolute_Coefficient'] = coef_df['Coefficient'].abs()
sorted_coef_df = coef_df.sort_values(by='Absolute_Coefficient', ascending=False)
#print(sorted_coef_df.head())

# Select the top 1,000 words
myvocab = sorted_coef_df['Feature'][:1000].tolist()

# Save myvocab to a file
with open('myvocab.txt', 'w') as file:
    for word in myvocab:
        file.write(word + '\n')

print(len(myvocab))
