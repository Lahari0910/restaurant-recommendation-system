from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Load and clean your dataset
df = pd.read_csv('zomato.csv')
df.rename(columns={'approx_cost(for two people)': 'cost'}, inplace=True)
df['rate'] = pd.to_numeric(df['rate'].astype(str).str.extract(r'(\d+\.\d+)')[0], errors='coerce')
df['cost'] = pd.to_numeric(df['cost'].astype(str).str.replace(',', ''), errors='coerce')
df['cuisines'] = df['cuisines'].fillna('')
df['name'] = df['name'].fillna('')

# Recommendation logic
def recommend(restaurant_name, df, top_n=10):
    target = df[df['name'].str.lower() == restaurant_name.lower()]
    if target.empty:
        return pd.DataFrame()

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
    similar = df[df['name'].str.lower() != restaurant_name.lower()]
    similar = similar.sort_values(by='score', ascending=False).head(top_n)
    return similar[['name', 'cuisines', 'rate', 'cost']]

# Homepage
@app.route('/')
def home():
    return render_template('landing.html')

# Form page
@app.route('/recommend-form')
def recommend_form():
    return render_template('form.html')

# Recommendation results
@app.route('/recommend', methods=['POST'])
def recommend_route():
    restaurant_name = request.form['restaurant']
    results = recommend(restaurant_name, df)
    return render_template('results.html', restaurant=restaurant_name, results=results)

# Visuals page
@app.route('/visuals')
def visuals():
    return render_template('visuals.html')

if __name__ == '__main__':
    app.run(debug=True)
