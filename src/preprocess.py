import pandas as pd
import numpy as np

def handle_missing_values(df):
    df['Marks'] = df['Marks'].fillna(df['Marks'].mean())
    df['StudyHours'] = df['StudyHours'].fillna(df['StudyHours'].median())
    return df

def remove_outliers(df):
    return df[(df['StudyHours'] <= 15) & (df['Marks'] <= 100)]

def feature_engineering(df):
    df['Performance'] = df['Marks'].apply(
        lambda x: 'Excellent' if x >= 80 else ('Good' if x >= 60 else 'Needs Improvement')
    )

    df['EffortScore'] = df['StudyHours'] * df['Attendance']

    df['AttendanceCategory'] = pd.cut(
        df['Attendance'],
        bins=[0, 60, 80, 100],
        labels=['Low', 'Medium', 'High']
    )

    return df