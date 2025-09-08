import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read the CSV file
df = pd.read_csv('cleaned_movies.csv')

# Fill NaN overviews with empty string
df['Overview'] = df['Overview'].fillna('')

# Compute TF-IDF matrix for movie overviews
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['Overview'])

# Compute cosine similarity matrix
similarity = cosine_similarity(tfidf_matrix)

# Save the similarity matrix as a pickle file
with open('similarity.pkl', 'wb') as f:
    pickle.dump(similarity, f)

print('similarity.pkl created successfully!')
