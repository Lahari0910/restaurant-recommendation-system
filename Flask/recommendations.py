import pandas as pd

# Load and clean the dataset
df = pd.read_csv('zomato.csv')
df['rate'] = pd.to_numeric(df['rate'].astype(str).str.extract(r'(\d+\.\d+)')[0], errors='coerce')
df['cost'] = pd.to_numeric(df['approx_cost(for two people)'].astype(str).str.replace(',', ''), errors='coerce')
df['cuisines'] = df['cuisines'].fillna('')

# Define the recommendation function
def recommend(restaurant_name, df, top_n=10):
    target = df[df['name'].str.lower() == restaurant_name.lower()]
    if target.empty:
        print(f"Restaurant '{restaurant_name}' not found.")
        return

    target_cuisines = set(str(target.iloc[0]['cuisines']).lower().split(', '))
    target_rating = target.iloc[0]['rate']
    target_cost = target.iloc[0]['cost']

    def similarity(row):
        cuisines = set(str(row['cuisines']).lower().split(', '))
        cuisine_score = len(target_cuisines & cuisines)
        rating_diff = abs(row['rate'] - target_rating)
        cost_diff = abs(row['cost'] - target_cost)
        return (cuisine_score * 2) - rating_diff - (cost_diff / 100)

    df['score'] = df.apply(similarity, axis=1)
    similar = df.sort_values(by='score', ascending=False).head(top_n)

    print(f"\nTOP {top_n} RESTAURANTS LIKE {restaurant_name.upper()} WITH SIMILAR REVIEWS:\n")
    for i, row in similar.iterrows():
        print(f"{row['name']} - Cuisines: {row['cuisines']} | Mean Rating: {row['rate']} | Cost: {row['cost']}")
recommend('Red Chilliez', df)
