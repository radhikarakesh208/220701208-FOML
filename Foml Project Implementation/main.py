import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('./clothing_dataset.csv')

brand_models = {}
brand_encoders = {}

feature_cols = ['Chest', 'Shoulder', 'Front_length', 'Sleeve_length']

for brand in df['Brand'].unique():
    brand_df = df[df['Brand'] == brand]
    
    X = brand_df[feature_cols]
    y = brand_df['Size']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    brand_encoders[brand] = le
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    brand_models[brand] = clf

model_data = {
    'models': brand_models,
    'encoders': brand_encoders
}

with open('model.pkl', 'wb') as file:
    pickle.dump(model_data, file)

user_chest = float(input("Enter chest size: "))
user_shoulder = float(input("Enter shoulder size: "))
user_front_length = float(input("Enter front length: "))
user_sleeve_length = float(input("Enter sleeve length: "))

user_input = pd.DataFrame([{
    'Chest': user_chest,
    'Shoulder': user_shoulder,
    'Front_length': user_front_length,
    'Sleeve_length': user_sleeve_length
}])

brands = ['Zara', 'H&M', 'Puma', 'Nike', 'Adidas'] 

for brand in brands:
    if brand not in brand_models:
        print(f"Model for brand '{brand}' not found.")
        continue

    model = brand_models[brand]
    encoder = brand_encoders[brand]
    
    predicted_size_num = model.predict(user_input)[0]
    
    predicted_size = encoder.inverse_transform([predicted_size_num])[0]
    
    print(f"Suggested Size for {brand}: {predicted_size}")

