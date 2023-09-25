import pandas as pd
import matplotlib.pyplot as plt
#importing data

data = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\Sem 5\MDCM\Investments_VC.csv", encoding="unicode_escape")

data.columns = data.columns.str.replace(' ', '_')
data.columns

data['city'] = data['city'].str.lower() 

data.rename(columns={'_funding_total_usd_': 'total_f'}, inplace=True)
data.rename(columns={'_market_': 'market'}, inplace=True)
data['total_f'] = pd.to_numeric(data['total_f'].str.replace(',', ''), errors='coerce')
data['total_f'].fillna(0, inplace=True)
# Fill NaN values with 0 for all numeric columns
numeric_columns = data.select_dtypes(include=['float64']).columns
data[numeric_columns] = data[numeric_columns].fillna(0)
mode_value = data['market'].mode()[0]
data['market'].fillna(mode_value, inplace=True)
freq_encoding = data['market'].value_counts(normalize=True)
data['market_encoded'] = data['market'].map(freq_encoding)
data['market_encoded'] 

country_counts = data.groupby('country_code').size()

data.fillna('Unknown', inplace=True)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
data['country_code_encoded'] = label_encoder.fit_transform(data['country_code'])
data['state_code_encoded'] = label_encoder.fit_transform(data['state_code'])

most_frequent_category = data['status'].mode()[0]
data['status'] = data['status'].replace("Unknown", most_frequent_category)

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

data['success'] = data['status'].apply(lambda status: 1 if status == 'operating' else (2 if status == 'acquired' else 0))

selected_features = [
    'total_f',
    'funding_rounds',
    'seed',
    'venture',
    'market',
    'debt_financing',
    'country_code',
    'state_code',
]

y = data['success']
X = data[selected_features]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50)

gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_classifier.fit(X_train, y_train)
prediction=gb_classifier.predict(X_test)
prediction

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pr=model.predict(X_test)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pr=model.predict(X_test)

def train_model_gbc(X,y):
    from sklearn.ensemble import GradientBoostingClassifier
    gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_classifier.fit(X_train, y_train)
    return gb_classifier

model=train_model_gbc(X_train,y_train)
with open('model.pkl','wb') as model_file:
    pickle.dump(model, model_file)

with open('preprocess_data.pkl','wb') as model_file:
    pickle.dump(preprocess_data, model_file)

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score

# %% [markdown]
# creating new Dataframe for Testing Data

# %%
testing_data=testing_data = {
    'total_f': [4000000.0],
    'funding_rounds': [2.0],
    'seed': [0.0],
    'venture': [4000000.0],
    #'angel': [0.0],
    #'private_equity': [0.0],
    'market_encoded': [0.02177],
   # 'equity_crowdfunding': [0.0],
    #'convertible_note': [0.0],
    'debt_financing': [0.0],
    'country_code_encoded': [110],
    'state_code_encoded': [6],
}

testing_df = pd.DataFrame(testing_data)
y_true= np.array([1])

# %%
testing_df

# %% [markdown]
# Loading our Model

# %%
with open('model.pkl','rb') as model_file:
    trained_model=pickle.load(model_file)
with open('preprocess_data.pkl','rb') as model_file:
    preprocess_data=pickle.load(model_file)

# %% [markdown]
# Testing our model with testing data

# %%
test=preprocess_data(testing_df)
y_pred=model.predict(test)
print(y_pred)

# %%
#1 is for Operating

# %% [markdown]
# Evaluation

# %%
# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


def preprocess_data(data):
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    scaler = StandardScaler()
    label_encoder = LabelEncoder()
    object_columns = data.select_dtypes(include=['object']).columns

    for column in object_columns:
        data[column] = label_encoder.fit_transform(data[column])

    scaled_features = scaler.fit_transform(data)

    data[data.columns] = scaled_features  

    return data

def train_model_gbc(X,y):
    from sklearn.ensemble import GradientBoostingClassifier
    gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_classifier.fit(X_train, y_train)
    return gb_classifier

model=train_model_gbc(X_train,y_train)
with open('model.pkl','wb') as model_file:
    pickle.dump(model, model_file)


with open('preprocess_data.pkl','wb') as model_file:
    pickle.dump(preprocess_data, model_file)

from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)


with open('model.pkl','rb') as model_file:
    model=pickle.load(model_file)
def preprocess_data(input_data):
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    scaler = StandardScaler()
    label_encoder = LabelEncoder()
    object_columns = input_data.select_dtypes(include=['object']).columns

    for column in object_columns:
        input_data[column] = label_encoder.fit_transform(input_data[column])

    scaled_features = scaler.fit_transform(input_data)

    #input_data[input_data.columns] = scaled_features  

    return scaled_features

@app.route('/')
def index():
    return render_template('index.html', predictions=None)

@app.route('/predict', methods=['POST'])
def predict():
    print('i am in predict function')
    if request.method == 'POST':
        total_f = float(request.form.get('total_f', 0.0))  # Handle missing field with a default value
        funding_rounds = float(request.form.get('funding_rounds', 0.0))  # Handle missing field with a default value
        seed = float(request.form.get('seed', 0.0))
        venture = float(request.form.get('venture',0.0))
        market_encoded = request.form.get('market_encoded',0.0)
        debt_financing = float(request.form.get('debt_financing', 0.0))
        country_code_encoded = request.form.get('country_code_encoded',0.0)
        state_code_encoded = request.form.get('state_code_encoded',0.0)
        
        # Create a DataFrame from the form data
        print('taking input')
        input_data = pd.DataFrame({
            'total_f': [total_f],
            'funding_rounds': [funding_rounds],
            'seed': [seed],
            'venture': [venture],
            'market_encoded': [market_encoded], 
            'debt_financing': [debt_financing],
            'country_code_encoded': [country_code_encoded],
            'state_code_encoded': [state_code_encoded]
        })

        preprocessed_df = preprocess_data(input_data)
        predictions = model.predict(preprocessed_df)
        print(predictions)
    

        return render_template('index.html', predictions=predictions[0])
       

if __name__ == '__main__':
    app.run()


