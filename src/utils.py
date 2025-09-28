import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import random

def train_model(path) :
    df = pd.read_csv(path)                                 
    print(df.head(), '\n\n')

    X = df.iloc[:, 0:-1]
    y = df.iloc[:, -1]

    non_numeric_cols = X.select_dtypes(include=['object']).columns
    print(f"Non-numeric col: {non_numeric_cols}", '\n\n')
    X = X.drop(columns=non_numeric_cols)

    from sklearn.preprocessing import StandardScaler
    num_cols = X.select_dtypes(include=['int32', 'int64', 'float32', 'float64']).columns
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
 
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(y_pred, "\n\n")

    # Accuracy
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy Score = {acc}", '\n\n')

    from sklearn.metrics import classification_report
    print('\n\n',"Classification Report:\n", classification_report(y_test, y_pred))

#----------------------------------------------------------------------------------------------------------------------------------------


def visualize_data(df) :

    # Histograms
    df.hist(figsize=(18, 18))
    plt.suptitle("Feature Distributions", fontsize=7)
    plt.show()

    # Counting labels
    plt.figure(figsize=(8, 8))
    sns.countplot(y='label', data=df, palette="Set1", hue='label', dodge=False)
    plt.title("Count of label data")
    plt.show() 


#---------------------------------------------------------------------------------------------------------------------------------

def generate_testcase(csv_path: str):
    """
    Generate a fake row (testcase) based on the schema of the dataset.
    Assumes the last column is the target (label).
    """
    df = pd.read_csv(csv_path)

    # Drop target column if exists (last one)
    features = df.iloc[:, :-1]  

    testcase = []
    for col in features.columns:
        if pd.api.types.is_numeric_dtype(features[col]):
            # numeric column → random value in min-max range
            min_val, max_val = features[col].min(), features[col].max()
            val = random.uniform(min_val, max_val)
            testcase.append(val)

        elif pd.api.types.is_categorical_dtype(features[col]) or features[col].dtype == object:
            # categorical column → pick random category
            val = random.choice(features[col].dropna().unique().tolist())
            testcase.append(val)

        elif pd.api.types.is_bool_dtype(features[col]):
            # boolean column → True/False
            val = random.choice([True, False])
            testcase.append(val)

        else:
            # fallback: string placeholder
            testcase.append("test_value")

    return testcase
