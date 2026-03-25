import pandas as pd

def generate_insights():
    # 1. بنقرأ الداتا النضيفة
    df = pd.read_csv('data_preprocessed.csv')

    # 2. المعلومة الأولى: متوسط عمر الطلاب في الداتا
    # (بما إننا عملنا Scaling للسن، هنحسبه من الداتا الأصلية أو نطلعه كنسبة)
    avg_age = df['Age at enrollment'].mean()
    insight1 = f"The average scaled age of students in this dataset is: {avg_age:.2f}"
    
    with open('insight1.txt', 'w') as f:
        f.write(insight1)

    # 3. المعلومة الثانية: نسبة الطلاب اللي عندهم ديون (Debtor)
    debtor_count = df['Debtor'].value_counts(normalize=True).get(1, 0) * 100
    insight2 = f"Percentage of students with outstanding debt: {debtor_count:.2f}%"
    
    with open('insight2.txt', 'w') as f:
        f.write(insight2)

    # 4. المعلومة الثالثة: متوسط درجات القبول (Admission grade)
    avg_grade = df['Admission grade'].mean()
    insight3 = f"The mean admission grade (scaled) across all students is: {avg_grade:.2f}"
    
    with open('insight3.txt', 'w') as f:
        f.write(insight3)

    print("✅ Step 4 Success: 3 Insights generated and saved as .txt files!")

if __name__ == "__main__":
    generate_insights()