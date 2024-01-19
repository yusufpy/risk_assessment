# Import necessary libraries
from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the pre-trained models and label encoder
df = pd.read_csv('risk_assessment.csv')

label_encoder = LabelEncoder()
df['Time of Day'] = label_encoder.fit_transform(df['Time of Day'])
df['Location'] = label_encoder.fit_transform(df['Location'])
df['Service Type'] = label_encoder.fit_transform(df['Service Type'])
df['Provider'] = label_encoder.fit_transform(df['Provider'])
df['Previous Transaction History'] = label_encoder.fit_transform(df['Previous Transaction History'])
df['Risk'] = label_encoder.fit_transform(df['Risk'])

X = df.drop(columns=['Risk'])
y = df['Risk']

model_amounts = RandomForestClassifier(n_estimators=100, random_state=42)
model_behavior = KNeighborsClassifier(n_neighbors=5)
model_history = LogisticRegression(random_state=42)

model_amounts.fit(X[['Transaction Amount', 'Historical Transaction Amount']], y)
model_behavior.fit(X[['Frequency']], y)
model_history.fit(X[['Time of Day', 'Location']], y)

ensemble_model = VotingClassifier(estimators=[
    ('amounts', model_amounts),
    ('behavior', model_behavior),
    ('history', model_history)
], voting='soft')

# Flask route to render the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Flask route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the HTML form
        transaction_amount = float(request.form['transaction_amount'])
        historical_transaction_amount = float(request.form['historical_transaction_amount'])
        frequency = int(request.form['frequency'])
        time_of_day = int(request.form['time_of_day'])
        location = 6#int(request.form['location'])

        # Make predictions using the ensemble model
        risk_score = (model_amounts.predict_proba([[transaction_amount, historical_transaction_amount]])[:, 1] +
                      model_behavior.predict_proba([[frequency]])[:, 1] +
                      model_history.predict_proba([[time_of_day, location]])[:, 1]) / 3
        
        if risk_score >= 0.16: # 'spam':
            b="..check the details"
        else:
            b="Good!"

        return render_template('index.html', risk_score=b)

if __name__ == '__main__':
    app.run(debug=True)

