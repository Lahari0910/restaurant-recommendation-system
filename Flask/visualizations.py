import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib
matplotlib.rcParams['font.family'] = 'Arial'  # or 'Noto Sans'


# Load dataset
zomato_df = pd.read_csv('zomato.csv')
df = pd.read_csv('zomato.csv')  # or use the correct path if it's in another folder
df['cuisines'] = df['cuisines'].astype(str).fillna('')


# Clean the 'rate' column
zomato_df['rate'] = zomato_df['rate'].astype(str).str.extract(r'(\d+\.\d+)')  # Extract numeric part
zomato_df['rate'] = pd.to_numeric(zomato_df['rate'], errors='coerce')         # Convert to float


# Top 6 Restaurants by Outlet Count
top_chains = zomato_df['name'].value_counts().head(6)
plt.figure(figsize=(10,6))
sns.barplot(x=top_chains.index, y=top_chains.values, palette='tab10')
plt.title("Top 6 Restaurants by Outlet Count in Bangalore")
plt.ylabel("Number of Outlets")
plt.xlabel("Restaurant Name")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('static/top_restaurants.png')
plt.close()

# Distribution of Restaurant Ratings
plt.figure(figsize=(10,5))
sns.histplot(zomato_df['rate'].dropna(), bins=20, color='skyblue')
plt.axvline(zomato_df['rate'].mean(), color='red', label='Mean')
plt.legend()
plt.title("Distribution of Restaurant Ratings")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig('static/rating_distribution.png')
plt.close()

# Top 10 Rated Restaurants
df_rating = zomato_df.drop_duplicates(subset='name')
df_rating['Mean Rating'] = df_rating.groupby('name')['rate'].transform('mean')
top_rated = df_rating.sort_values('Mean Rating', ascending=False).drop_duplicates('name').head(10)

plt.figure(figsize=(10,8))
sns.barplot(data=top_rated, x='Mean Rating', y='name', palette='hls')
plt.title("Top 10 Rated Restaurants in Bangalore")
plt.xlabel("Mean Rating")
plt.ylabel("Restaurant Name")
plt.tight_layout()
plt.savefig('static/top_rated.png')
plt.close()
def get_top_words(text_series, top_n=15, ngram_range=(2,2)):
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english')
    X = vectorizer.fit_transform(text_series)
    word_counts = X.sum(axis=0).A1
    vocab = vectorizer.get_feature_names_out()
    word_freq = list(zip(vocab, word_counts))
    word_freq.sort(key=lambda x: x[1], reverse=True)
    return word_freq[:top_n]
# Top 15 two-word frequencies for cuisines
top_bigrams = get_top_words(df['cuisines'], top_n=15, ngram_range=(2,2))
df_bigrams = pd.DataFrame(top_bigrams, columns=['Word', 'Count'])

plt.figure(figsize=(10,7))
sns.barplot(data=df_bigrams, x='Count', y='Word', palette='mako')
plt.title('Word Couple Frequency for Cuisines')
plt.tight_layout()
plt.savefig('static/cuisine_bigrams.png')
plt.close()

