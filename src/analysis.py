def show_basic(df):
    print("\n📊 Basic Statistics:\n", df.describe())

def top_students(df):
    print("\n🏆 Top Students:\n", df.nlargest(5, 'Marks'))

def low_students(df):
    print("\n⚠️ Low Performers:\n", df[df['Marks'] < 50])

def group_analysis(df):
    print("\n📊 Average Marks by Attendance Category:")
    print(df.groupby('AttendanceCategory')['Marks'].mean())