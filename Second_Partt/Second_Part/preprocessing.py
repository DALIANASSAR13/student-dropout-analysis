import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess(file_path):
    # بنقرأ الداتا اللي طلعت من الـ ingest
    df = pd.read_csv(file_path)

    # 1. Data Cleaning (تنظيف)
    df.drop_duplicates(inplace=True) # مسح المكرر
    df.fillna(df.mean(numeric_only=True), inplace=True) # تعبئة أي قيم ناقصة بالمتوسط

    # 2. Feature Transformation (تحويل)
    le = LabelEncoder()
    df['Target'] = le.fit_transform(df['Target']) # تحويل Dropout/Graduate لأرقام
    
    scaler = StandardScaler()
    # تحجيم درجات القبول وسن الطالب
    cols_to_scale = ['Admission grade', 'Age at enrollment']
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    # 3. Discretization (تقسيم فئات)
    # هنقسم السن لـ 3 مجموعات (صغير، متوسط، كبير)
    df['Age_Bins'] = pd.cut(df['Age at enrollment'], bins=3, labels=[0, 1, 2])

    # 4. Dimensionality Reduction (تقليل الأبعاد)
    # هنختار أهم 5 أعمدة بس عشان الموديل ميتوهش
    important_cols = ['Target', 'Admission grade', 'Age at enrollment', 'Debtor', 'Scholarship holder']
    df_final = df[important_cols]

    # حفظ النتيجة
    df_final.to_csv('data_preprocessed.csv', index=False)
    print("✅ Step 2 Success: Preprocessing complete!")

if __name__ == "__main__":
    preprocess(sys.argv[1])