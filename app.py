import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
import joblib



df = pd.read_csv('survey.csv')

print(df.describe())

print(df.info())


# EDA

# unique value for every column to check 
# for i in  df :
#     print(f'{i} -  unique vaules = {df[i].unique()}')

# age range - 18 - 75
df = df[(df['Age'] >= 18) & (df['Age'] <= 75)]

# gender - categorised
def clean_gender(g):
    g = str(g).strip().lower()

    male_keywords = ['male', 'm', 'man', 'msle', 'mal', 'maile', 'male-ish', 'cis male', 'cis man', 'make', 'malr', 'mail']
    female_keywords = ['female', 'f', 'woman', 'cis female', 'femake', 'femail', 'female ', 'female (cis)']
    other_keywords = ['trans', 'non-binary', 'genderqueer', 'fluid', 'androgyne', 'agender', 'enby', 'neuter', 'queer', 'nah', 'all', 'p', 'a little about you']

    if any(word in g for word in male_keywords):
        return 'Male'
    elif any(word in g for word in female_keywords):
        return 'Female'
    else:
        return 'Other'

df['Gender'] = df['Gender'].apply(clean_gender)

# self employes handeling missing values
df['self_employed'] = df['self_employed'].fillna('Unknown')

# work-interferance missing values
df['work_interfere'] = df['work_interfere'].fillna('Unknown')

#drop columns
df = df.drop(['Timestamp' ], axis = 1)


print(df.info())

# classification

def classification(df_class):
    # grouping the ages fro a bettere classification
    bins = [18, 25, 35, 45, 60, 75]
    labels = ['18–25', '26–35', '36–45', '46–60', '61–75']
    df_class['age_group'] = pd.cut(df_class['Age'], bins=bins, labels=labels, include_lowest=True)
    df_class['age_group'] = df_class['age_group'].astype(str)


    # cateoring data
    X = df_class.drop(['comments','treatment' , 'Age'] , axis =1)
    y = df_class['treatment'].map({'Yes': 1, 'No': 0})

    # spliting data
    X_train , X_test , y_train, y_test = train_test_split(X,y,test_size = 0.2 , random_state = 42)
    
    #  categoring it 
    categorical_cols = X_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # encoding the data
    encoder = ColumnTransformer(
        transformers=[
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols),
    ]
    )

    X_train_encoded = encoder.fit_transform(X_train)
    X_test_encoded = encoder.transform(X_test)

    

    
    param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, None],
    'min_samples_split': [2, 5]
     }

    
    rf = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,         # use all cores
    verbose=2,
    scoring='accuracy' # or 'f1' if you prefer
    )

    grid_search.fit(X_train_encoded, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_encoded)

    joblib.dump(best_model, 'classi_model.pkl')
    joblib.dump(encoder, 'classify_preprocessor.pkl')



    # evaluating data

    print("✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))
    print("✅ Accuracy Score:", accuracy_score(y_test, y_pred))




classification(df)