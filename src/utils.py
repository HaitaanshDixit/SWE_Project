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
'''
def generate_testcase(csv_path):
    df = pd.read_csv(csv_path)

    # Drop target column if exists
    target_col = "label"  # <-- replace with your actual target
    if target_col in df.columns:
        features = df.drop(columns=[target_col])
    else:
        features = df

    # Pick one row (or synthesize)
    row = features.sample(1, random_state=42).iloc[0]

    # Force numeric
    #row = pd.to_numeric(row, errors="coerce").fillna(0)

    return row.to_numpy().flatten()

def generate_testcase(csv_path):
    df = pd.read_csv(csv_path)

    target_col = "label"  # <-- replace with your target column name
    if target_col in df.columns:
        features = df.drop(columns=[target_col])
    else:
        features = df

    # Sample a single row (keep as DataFrame, not Series)
    row = features.sample(1, random_state=random.randint(0, 1000))

    # Return it as dictionary (preserves column names and types)
    return row.to_dict(orient="records")[0]
'''

def generate_testcase(csv_path):
    df = pd.read_csv(csv_path)

    target_col = "label"  
    if target_col in df.columns:
        features = df.drop(columns=[target_col])
    else:
        features = df

    features = features.select_dtypes(include=[np.number])

    testcase = {}
    for col in features.columns:
        mean = features[col].mean()
        std = features[col].std()
        if pd.isna(std) or std == 0:
            std = 1  
        testcase[col] = float(np.random.normal(mean, std))

    return testcase