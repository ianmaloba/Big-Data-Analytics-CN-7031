# Big Data Network Intrusion Detection System (UNSW-NB15)

This project demonstrates how modern big data technologies can be used to build an effective **network intrusion detection system (IDS)**.  
The analysis uses **Apache Hive** for distributed data exploration and **Apache Spark (PySpark)** for statistical analysis and machine learning.

Using the **UNSW-NB15 dataset (~2.5M network flows)**, the system analyzes network behavior and trains models capable of detecting malicious traffic patterns.

## Full Technical Walkthrough

This repository only contains the code and results. A full explanation of the methodology, analysis process, Hive queries, and ML results is available in the blog post:
https://ianmaloba.com/blogs/article/building-a-network-intrusion-detection-system-with-big-data-apache-hive-and-pyspark-on-unsw-nb15/

## Example Model Output

Binary classification confusion matrix:

![Binary Confusion Matrix](https://raw.githubusercontent.com/ianmaloba/Big-Data-Analytics-CN-7031/main/results/pyspark_results/binary_confusion_matrix.png)

## Key Results

- **Binary Classification (Normal vs Attack):**
  - Accuracy: **99.07%**
  - AUC-ROC: **99.95%**
  - F1-Score: **99.08%**

- **Multi-class Classification (Attack Type Detection):**
  - Accuracy: **97.79%**
  - Weighted F1-Score: **97.33%**

The models successfully identify malicious network behavior across multiple attack categories.

## Dataset

This project uses the **UNSW-NB15 Network Intrusion Dataset**.

Download the dataset (581MB):

https://www.dropbox.com/s/4xqg32ih9xoh5jq/UNSW-NB15.csv?dl=1

After downloading, place the file in: data/UNSW-NB15.csv


## Repository Contents

- **Scripts (Hive + PySpark analysis):**  
  https://github.com/ianmaloba/Big-Data-Analytics-CN-7031/tree/main/scripts

- **Dataset metadata and feature descriptions:**  
  https://github.com/ianmaloba/Big-Data-Analytics-CN-7031/tree/main/data

- **All generated results and visualizations:**  
  https://github.com/ianmaloba/Big-Data-Analytics-CN-7031/tree/main/results


## Contact

If you have questions or suggestions, feel free to contact me at [**contact@ianmaloba.com**](mailto:contact@ianmaloba.com) or open an issue on this repo.
