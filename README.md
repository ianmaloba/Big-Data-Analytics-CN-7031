# Big Data Analytics: Network Intrusion Detection System

This repository contains one of my assignments completed as part of my coursework. The dataset used is the UNSW-NB15.

The code and analysis are available for educational reference purposes...

## Contact

For questions or discussions about this project, please feel free to reach out through GitHub issues.

## Project Overview

I implemented a complete big data analytics pipeline for network intrusion detection, utilizing Apache Hive for distributed query processing and Apache Spark with PySpark for advanced statistical analysis and machine learning. The analysis encompasses exploratory data analysis, statistical characterization, and the development of both binary and multi-class classification models to detect and categorize network attacks.

The project achieved exceptional results with 99.07% accuracy for binary classification (normal vs attack traffic) and 97.68% accuracy for multi-class classification across 14 attack categories. These results demonstrate the effectiveness of modern machine learning approaches for cybersecurity applications.

## Dataset

### UNSW-NB15 Network Intrusion Dataset

The UNSW-NB15 dataset was created by the Australian Centre for Cyber Security (ACCS) and contains a hybrid of real modern normal activities and synthetic contemporary attack behaviours. The dataset includes approximately 2.5 million network flow records with 49 features extracted from raw network packets.

**Dataset Source:** [UNSW-NB15 Dataset](https://www.dropbox.com/s/4xqg32ih9xoh5jq/UNSW-NB15.csv?dl=1) (581MB CSV file)

**Attack Categories:** The dataset includes nine attack types:
- Fuzzers
- Analysis
- Backdoors
- Denial of Service (DoS)
- Exploits
- Generic
- Reconnaissance
- Shellcode
- Worms

**Feature Documentation:** [UNSW-NB15 Features Description](./data/UNSW-NB15_features.csv)

**Attack Subcategories:** [UNSW-NB15 Event List](./data/UNSW-NB15_LIST_EVENTS.csv)

## Setup Instructions

### 1. Dataset Download
**IMPORTANT:** The main dataset file is not included in this repository due to its large size (580MB). You must download it manually:

1. **Download the dataset** from: [UNSW-NB15 Dataset](https://www.dropbox.com/s/4xqg32ih9xoh5jq/UNSW-NB15.csv?dl=1)
2. **Save the file as:** `UNSW-NB15.csv`
3. **Place it in the:** `data/` folder
4. **Verify the file structure:**
   ```
   data/
   ├── UNSW-NB15.csv                 ← Download and place here
   ├── UNSW-NB15_features.csv        ← Already included
   └── UNSW-NB15_LIST_EVENTS.csv     ← Already included
   ```

### 2. Environment Setup
```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # On Windows
# source .venv/bin/activate  # On macOS/Linux

# Install dependencies (compatible versions for Windows)
pip install -r requirements.txt
```

### 3. Verify Setup
```bash
# Test if PySpark is working
python -c "import pyspark; print('PySpark version:', pyspark.__version__)"

# Check dataset
python -c "import os; print('Dataset exists:', os.path.exists('data/UNSW-NB15.csv'))"
```

## Assignment Tasks

The coursework was structured around five main tasks:

### Task 1: Dataset Understanding
I analyzed the UNSW-NB15 dataset structure, understanding its 49 features including temporal characteristics, statistical metrics, connection patterns, and protocol information. The dataset exhibits severe class imbalance with 87.35% normal traffic and 12.65% attack traffic, requiring careful consideration in model development.

### Task 2: Big Data Query and Analysis using Apache Hive
I developed four complex Hive queries utilizing built-in functions for data exploration:

**Query 1 - Attack Distribution Analysis:**
This query employed CASE statements and aggregation functions (COUNT, AVG, ROUND) to analyze traffic distribution across attack categories. The results revealed that Generic attacks are most prevalent with 215,481 instances, while Exploits demonstrate the longest average duration at 2.12 seconds.

**Visualizations:**
- [Traffic Distribution Pie Chart](./results/hive_visualizations/q1_traffic_distribution_pie.png)
- [Attack Categories Bar Chart](./results/hive_visualizations/q1_attack_categories_bar.png)
- [Average Bytes Comparison](./results/hive_visualizations/q1_avg_bytes_comparison.png)

**Query 2 - Protocol and Service Analysis:**
Using string functions (UPPER) and conditional aggregation, this query identified critical protocol vulnerabilities. I discovered that SCTP and UNAS protocols exhibit 100% attack rates, and TCP/POP3 shows 99.74% attack rate, indicating severe security concerns for email protocols.

**Visualizations:**
- [Attack Percentage by Protocol](./results/hive_visualizations/q2_attack_percentage_by_protocol.png)
- [Connection vs Attack Scatter Plot](./results/hive_visualizations/q2_connection_vs_attack_scatter.png)
- [Top 5 Protocols Stacked Chart](./results/hive_visualizations/q2_top5_protocols_stacked.png)

**Query 3 - Connection State Statistical Analysis:**
This query utilized advanced statistical functions including STDDEV, PERCENTILE, MIN, and MAX to analyze traffic patterns across connection states. The analysis revealed that intermediate (INT) connection states show the highest vulnerability with 53.14% attack rate.

**Visualizations:**
- [Attacks by State](./results/hive_visualizations/q4_attacks_by_state.png)
- [State Statistics Multi-panel](./results/hive_visualizations/q4_state_statistics_multi.png)
- [Bytes Distribution by State](./results/hive_visualizations/q4_bytes_distribution.png)

**Query Results Data:**
- [Query 1 Results](./results/hive_results/query1/000000_0)
- [Query 2 Results](./results/hive_results/query2/000000_0)
- [Query 4 Results](./results/hive_results/query4/000000_0)
- [Summary Statistics](./results/hive_visualizations/summary_statistics.txt)

### Task 3: Advanced Analytics using PySpark

#### 3.1 Statistical Analysis
I performed comprehensive statistical analysis using PySpark DataFrames:

- **Descriptive Statistics:** Computed mean, standard deviation, min, and max values for 13 numerical features, revealing highly skewed distributions with duration mean of 0.77s but standard deviation of 8.34s.

- **Correlation Analysis:** Generated correlation matrices on a 10% stratified sample, identifying weak individual correlations but strong multivariate relationships between features.

- **Distribution Analysis:** Created histograms comparing feature distributions between normal and attack traffic, showing distinct patterns in duration, transfer rates, and byte counts.

- **Attack Category Analysis:** Analyzed the distribution of attack types, identifying severe class imbalance within attack categories.

**Statistical Analysis Results:**
- [Descriptive Statistics CSV](./results/pyspark_results/descriptive_statistics.csv)
- [Correlation Matrix CSV](./results/pyspark_results/correlation_matrix.csv)
- [Summary Statistics CSV](./results/pyspark_results/part1_summary.csv)

**Visualizations:**
- [Descriptive Statistics Table](./results/pyspark_results/stats_descriptive_table.png)
- [Label Distribution](./results/pyspark_results/stats_label_distribution.png)
- [Correlation Heatmap](./results/pyspark_results/stats_correlation_heatmap.png)
- [Feature Distributions](./results/pyspark_results/stats_feature_distributions.png)
- [Attack Categories Distribution](./results/pyspark_results/stats_attack_categories.png)

#### 3.2 Binary Classification
I implemented a Random Forest classifier for binary classification (normal vs attack):

**Model Configuration:**
- Algorithm: Random Forest
- Number of trees: 100
- Maximum depth: 10
- Features: 26 selected features
- Train/Test split: 70/30

**Performance Metrics:**
- Accuracy: 99.07%
- AUC-ROC: 99.95%
- F1-Score: 99.08%
- Precision: 99.08%
- Recall: 99.07%
- False Positive Rate: 0.66%
- Attack Detection Rate: 97.20%

The model achieved near-perfect class separation with an AUC-ROC score approaching the theoretical maximum of 1.0, indicating excellent discriminative ability.

**Binary Classification Results:**
- [Classification Metrics CSV](./results/pyspark_results/binary_classification_metrics.csv)
- [Classification Report CSV](./results/pyspark_results/binary_classification_report.csv)
- [Feature Importance CSV](./results/pyspark_results/binary_feature_importance.csv)

**Visualizations:**
- [Confusion Matrix](./results/pyspark_results/binary_confusion_matrix.png)
- [ROC Curve](./results/pyspark_results/binary_roc_curve.png)
- [Feature Importance](./results/pyspark_results/binary_feature_importance.png)
- [Metrics Comparison](./results/pyspark_results/binary_metrics_comparison.png)

#### 3.3 Multi-class Classification
I extended the analysis to multi-class classification identifying specific attack types:

**Model Configuration:**
- Algorithm: Random Forest
- Classes: 14 (1 Normal + 13 Attack categories including duplicates due to inconsistent labelling)
- Same architecture as binary classifier

**Performance Metrics:**
- Overall Accuracy: 97.68%
- Weighted F1-Score: 97.11%
- Weighted Precision: 97.67%
- Weighted Recall: 97.68%

**Per-Class Performance:**
- High Performance (F1 > 0.95): Normal, Generic, Exploits
- Moderate Performance (F1: 0.70-0.85): Reconnaissance, Fuzzers
- Challenging (F1 < 0.65): Worms, Shellcode, Backdoors

The degraded performance for rare attack types reflects the challenge of learning from severely imbalanced data where some classes have fewer than 200 training examples.

**Multi-class Classification Results:**
- [Multi-class Metrics CSV](./results/pyspark_results/multiclass_metrics.csv)
- [Classification Report CSV](./results/pyspark_results/multiclass_classification_report.csv)
- [Feature Importance CSV](./results/pyspark_results/multiclass_feature_importance.csv)
- [Class Distribution CSV](./results/pyspark_results/multiclass_class_distribution.csv)

**Visualizations:**
- [Class Distribution](./results/pyspark_results/multiclass_distribution.png)
- [Confusion Matrix](./results/pyspark_results/multiclass_confusion_matrix.png)
- [Normalized Confusion Matrix](./results/pyspark_results/multiclass_confusion_normalized.png)
- [Performance Metrics](./results/pyspark_results/multiclass_performance_metrics.png)
- [Feature Importance](./results/pyspark_results/multiclass_feature_importance.png)
- [Overall Metrics](./results/pyspark_results/multiclass_overall_metrics.png)

## Scripts and Code

### PySpark Part 1: Statistical Analysis
**File:** [pyspark_part1_statistical_analysis.py](./scripts/pyspark_part1_statistical_analysis.py)

This script performs comprehensive statistical analysis on the UNSW-NB15 dataset including:
- Descriptive statistics computation for 13 numerical features
- Label distribution analysis
- Correlation matrix generation
- Feature distribution visualization
- Attack category distribution analysis

**Outputs:** 8 files (7 visualizations + 1 CSV summary)

### PySpark Part 2: Binary Classification
**File:** [pyspark_part2_binary_classification.py](./scripts/pyspark_part2_binary_classification.py)

This script implements binary classification for intrusion detection:
- Feature engineering with 26 selected features
- Random Forest model training (100 trees, max depth 10)
- Model evaluation with multiple metrics
- Confusion matrix and ROC curve generation
- Feature importance analysis

**Outputs:** 7 files (4 visualizations + 3 CSV reports)

### PySpark Part 3: Multi-class Classification
**File:** [pyspark_part3_multiclass_classification.py](./scripts/pyspark_part3_multiclass_classification.py)

This script extends classification to identify specific attack types:
- Multi-class label preparation and indexing
- Random Forest training for 14 categories
- Per-class performance evaluation
- Confusion matrix visualization (normalized and absolute)
- Class-specific metrics analysis

**Outputs:** 10 files (6 visualizations + 4 CSV reports)

### Hive Visualization Script
**File:** [visualize_hive.py](./scripts/visualize_hive.py)

This script generates visualizations from Hive query results:
- Reads exported query results from CSV files
- Creates professional charts and graphs
- Saves visualizations as high-resolution PNG files

**Outputs:** 9 visualizations + 1 summary text file

## Key Findings

### Feature Importance Discovery
I discovered that connection pattern features (ct_state_ttl, ct_srv_dst, ct_dst_ltm) proved most discriminative for attack detection, accounting for over 39% of total feature importance. This suggests that attackers exhibit characteristic behavioral patterns in how they establish connections rather than simply transmitting unusual volumes or types of data.

### Protocol Vulnerabilities
My Hive analysis identified critical vulnerabilities in specific protocol-service combinations. SCTP and UNAS protocols showed 100% attack rates, while TCP/POP3 exhibited 99.74% attack rate across 1,533 connections. Most concerning was the UDP/DNS service with 210,566 malicious connections representing 26.94% attack rate, highlighting a major security concern for fundamental internet infrastructure.

### Connection State Exploitation
The analysis revealed that intermediate (INT) connection states are most vulnerable to attacks, with 53.14% of INT connections being malicious. This indicates attackers frequently exploit the connection establishment phase before connections are fully established, suggesting the importance of monitoring partial connection states in intrusion detection systems.

### Class Imbalance Impact
While binary classification achieved exceptional performance despite severe class imbalance, multi-class classification showed clear performance degradation for rare attack types. This demonstrates the need for specialized techniques such as SMOTE oversampling or class-weighted loss functions for operational systems requiring detection of all attack types.

## Performance Summary

| Task | Metric | Value |
|------|--------|-------|
| Binary Classification | Accuracy | 99.07% |
| Binary Classification | AUC-ROC | 99.95% |
| Binary Classification | F1-Score | 99.08% |
| Binary Classification | False Positive Rate | 0.66% |
| Multi-class Classification | Accuracy | 97.68% |
| Multi-class Classification | Weighted F1-Score | 97.11% |
| Dataset Size | Total Records | 2,539,739 |
| Dataset Size | Features | 49 |
| Training Time (Binary) | Approximate | 8-10 minutes |
| Training Time (Multi-class) | Approximate | 10-12 minutes |

## Installation and Usage

### Prerequisites
```bash
# Install Java 17
brew install openjdk@17

# Set JAVA_HOME
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
export PATH=$JAVA_HOME/bin:$PATH

# Verify Java installation
java -version
```

### Python Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate  # On Windows
# source .venv/bin/activate  # On macOS/Linux

# Install compatible packages (Windows-tested versions)
pip install -r requirements.txt
```

**Note:** The requirements.txt includes specific package versions tested for Windows compatibility, including PySpark 3.5.1 with findspark for better Windows support.

### Running the Analysis

**Important:** Ensure the UNSW-NB15.csv dataset is downloaded and placed in the `data/` folder before running any PySpark scripts.

#### Statistical Analysis
```bash
python scripts/pyspark_part1_statistical_analysis.py
```

#### Binary Classification
```bash
python scripts/pyspark_part2_binary_classification.py
```

#### Multi-class Classification
```bash
python scripts/pyspark_part3_multiclass_classification.py
```

#### Hive Visualization
```bash
python scripts/visualize_hive.py
```

## Academic References

The project utilized some contemporary research and industry best practices:

- Moustafa, N. and Slay, J. (2015) 'UNSW-NB15: a comprehensive data set for network intrusion detection systems', Military Communications and Information Systems Conference (MilCIS).

- Breiman, L. (2001) 'Random Forests', Machine Learning.

- Zaharia, M., Chowdhury, M., Franklin, M.J., Shenker, S. and Stoica, I. (2010) 'Spark: Cluster Computing with Working Sets', HotCloud.

- Thusoo, A., et al. (2009) 'Hive: a warehousing solution over a map-reduce framework', Proceedings of the VLDB Endowment.

## Repository Structure
```
Big Data Analytics (CN-7031)
├─ data
│  ├─ UNSW-NB15_features.csv
│  └─ UNSW-NB15_LIST_EVENTS.csv
├─ README.md
├─ requirements.txt
├─ results
│  ├─ hive_results
│  │  ├─ query1
│  │  │  └─ 000000_0
│  │  ├─ query2
│  │  │  └─ 000000_0
│  │  └─ query4
│  │     └─ 000000_0
│  ├─ hive_visualizations
│  │  ├─ q1_attack_categories_bar.png
│  │  ├─ q1_avg_bytes_comparison.png
│  │  ├─ q1_traffic_distribution_pie.png
│  │  ├─ q2_attack_percentage_by_protocol.png
│  │  ├─ q2_connection_vs_attack_scatter.png
│  │  ├─ q2_top5_protocols_stacked.png
│  │  ├─ q4_attacks_by_state.png
│  │  ├─ q4_bytes_distribution.png
│  │  ├─ q4_state_statistics_multi.png
│  │  └─ summary_statistics.txt
│  └─ pyspark_results
│     ├─ binary_classification_metrics.csv
│     ├─ binary_classification_report.csv
│     ├─ binary_confusion_matrix.png
│     ├─ binary_feature_importance.csv
│     ├─ binary_feature_importance.png
│     ├─ binary_metrics_comparison.png
│     ├─ binary_roc_curve.png
│     ├─ correlation_matrix.csv
│     ├─ descriptive_statistics.csv
│     ├─ multiclass_classification_report.csv
│     ├─ multiclass_class_distribution.csv
│     ├─ multiclass_confusion_matrix.png
│     ├─ multiclass_confusion_normalized.png
│     ├─ multiclass_distribution.png
│     ├─ multiclass_feature_importance.csv
│     ├─ multiclass_feature_importance.png
│     ├─ multiclass_metrics.csv
│     ├─ multiclass_overall_metrics.png
│     ├─ multiclass_performance_metrics.png
│     ├─ part1_summary.csv
│     ├─ stats_attack_categories.png
│     ├─ stats_correlation_heatmap.png
│     ├─ stats_descriptive_table.png
│     ├─ stats_feature_distributions.png
│     └─ stats_label_distribution.png
└─ scripts
   ├─ pyspark_part1_statistical_analysis.py
   ├─ pyspark_part2_binary_classification.py
   ├─ pyspark_part3_multiclass_classification.py
   └─ visualize_hive.py
```