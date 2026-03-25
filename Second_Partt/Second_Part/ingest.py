import pandas as pd
import sys

def start_ingest(file_path):
    # بنقرأ الملف ونعرفه إن الفاصل هو ";"
    data = pd.read_csv(file_path, sep=';')
    
    # بنسيف نسخة تانية اسمها data_raw.csv
    
    data.to_csv('data_raw.csv', index=False)
    
    print("✅ Success! Raw data is ready.")

if __name__ == "__main__":
    # بنقول للكود يقرأ اسم الملف اللي هنكتبه في التيرمينال
    start_ingest(sys.argv[1])