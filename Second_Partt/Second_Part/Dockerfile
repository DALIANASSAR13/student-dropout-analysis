# 1. بنحدد نسخة البايثون اللي هنبدأ بيها (القاعدة)
FROM python:3.11-slim

# 2. بننزل المكتبات اللي الأساينمنت طلبها
RUN pip install --no-cache-dir pandas numpy matplotlib seaborn scikit-learn scipy requests

# 3. بنعمل فولدر جوه دوكر عشان نحط فيه كودنا
RUN mkdir -p /app/pipeline/

# 4. بننسخ كل ملفات المشروع (.py و .csv) من جهازك لجوه دوكر
COPY . /app/pipeline/

# 5. بنقول لدوكر "خليك واقف جوه الفولدر ده" أول ما تفتح
WORKDIR /app/pipeline/

# 6. ده الجزء اللي كان ناقصك: بنفتح شاشة الأوامر (bash) أول ما نشغل الصندوق
CMD ["/bin/bash"]