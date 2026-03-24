
# student-dropout-analysis

## Course
CSCI461 – Introduction to Big Data  
Assignment #1 

---

# Project Overview

This project implements a **data processing pipeline inside a Docker container** to analyze student academic performance and predict dropout behavior using the dataset **"Predict Students' Dropout and Academic Success"**.

The pipeline performs multiple stages including:

- Data ingestion
- Data preprocessing
- Data analysis
- Data visualization
- Clustering

All stages are executed inside a Docker environment to ensure reproducibility.

Dataset Source:  
UCI Machine Learning Repository  
https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

---

# Project Structure

```
customer-analytics/
│
├── Dockerfile
├── ingest.py
├── preprocess.py
├── analytics.py
├── visualize.py
├── cluster.py
├── summary.sh
├── README.md
└── results/
```

---

# File Description

| File | Description |
|-----|-------------|
| Dockerfile | Builds the Docker container and installs required libraries |
| ingest.py | Loads the dataset and saves a raw copy |
| preprocess.py | Cleans and transforms the dataset |
| analytics.py | Generates textual insights |
| visualize.py | Creates data visualizations |
| cluster.py | Applies K-Means clustering |
| summary.sh | Copies generated results and stops the container |
| results/ | Stores generated output files |

---

# Libraries Used

The project uses the following Python libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy
- requests

These libraries are used for data manipulation, visualization, and machine learning.

---

# Execution Flow

The pipeline runs in the following order:

Dataset  
↓  
ingest.py  
↓  
preprocess.py  
↓  
analytics.py  
↓  
visualize.py  
↓  
cluster.py  
↓  
results

Each script calls the next stage and passes the processed dataset as input.

---

# Build Docker Image

Run the following command in the project directory:

```bash
docker build -t customer-analytics .
```

---

# Run Docker Container

```bash
docker run -it customer-analytics
```

---

# Run the Pipeline

Inside the container run:

```bash
python ingest.py dataset.csv
```

This will execute the full pipeline.

---

# Output Files

The pipeline generates the following outputs:

```
data_raw.csv
data_preprocessed.csv
insight1.txt
insight2.txt
insight3.txt
summary_plot.png
clusters.txt
```

All outputs are stored in the `results/` folder.

---

# Team Members

Add your team members here:

- Dalia Nassar 231001310
- Maryam Mahmoud 231000399
- AlaaAllah arafa 231000568

---


