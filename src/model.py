from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

def train_model(df):
    X = df[['StudyHours', 'Attendance']]
    y = df['Marks']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"\n📈 Model Accuracy (R²): {round(score,2)}")

    return model

def predict_marks(model, study_hours, attendance):
    input_df = pd.DataFrame(
        [[study_hours, attendance]],
        columns=['StudyHours', 'Attendance']
    )
    return model.predict(input_df)[0]