# This is data preprocessing script for the heart disease prediction project. It reads the raw data, performs necessary cleaning and transformations, and saves the processed data for further analysis and modeling.
import pandas as pd
from sklearn.model_selection import train_test_split    # For splitting the data into training and testing sets
from sklearn.preprocessing import StandardScaler        # For feature scaling 

def preprocess_data(file_path="D:\\Heart_disease_predicton_pytorch\\data\\processed\\heart_disease_data.csv"):
    # Read the processed data
    df = pd.read_csv(file_path)
    
    # Separate features and target variable
    X = df.drop("target", axis=1)  # Features
    y = df["target"]  # Target variable

    # Train-test split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=45)

    #split test data into validation and test sets (50% validation, 50% test)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test, random_state=45)
    
    # Initialize the scaler
    scaler = StandardScaler() #scales the features to have mean=0 and std=1, which is important for many machine learning algorithms to perform well.
    
    # Fit the scaler on the training data and transform all sets
    X_train_scaled = scaler.fit_transform(X_train)  #Fit the scaler to the training data and transform it
    X_val_scaled = scaler.transform(X_val)          #Transform the validation data using the same scaler (without fitting)
    X_test_scaled = scaler.transform(X_test)        #Transform the test data using the same scaler (without fitting)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test