<div align="center">

# 📊 EDA Tool

### Automated Exploratory Data Analysis & Cleaning — powered by Streamlit

<br/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Built with Claude](https://img.shields.io/badge/Built%20with-Claude%20AI-D97706?style=for-the-badge&logo=anthropic&logoColor=white)](https://claude.ai/)

<br/>

> 🤖 **Note:** This project was built with the assistance of Claude AI (Anthropic). The logic, structure, and features were designed and iterated by the developer — Claude helped with code generation and debugging.

<br/>

</div>

---

## What is this?

**EDA Tool** is a no-code data analysis app. Drop in a CSV or Excel file and instantly get a health score, visualizations, missing value reports, and a full data cleaning pipeline — all in your browser.

No Jupyter. No scripts. Just upload and go.

---

## Features

<br/>

<table>
<tr>
<td width="50%">

### 📊 Dataset Health Score
Automatic 0–100 quality score based on missing values, duplicates, constant columns, and outlier density. Colour-coded: Excellent / Good / Needs Cleaning / Poor.

</td>
<td width="50%">

### 🗂️ Feature Summary
Classifies every column automatically — Numeric, Categorical, Boolean, Datetime, Object — displayed as metric cards at a glance.

</td>
</tr>
<tr>
<td width="50%">

### 🔍 Missing Value Analysis
Visual horizontal bar chart showing missing counts per column, sorted by severity. Green success message when your data is clean.

</td>
<td width="50%">

### 📈 Distribution Analysis
Histogram + box plot overlay for any numeric column. Pick a column, see its full distribution instantly.

</td>
</tr>
<tr>
<td width="50%">

### 🔗 Correlation Heatmap
Interactive Plotly heatmap with selectable method — **Pearson**, **Spearman**, or **Kendall** — updates dynamically.

</td>
<td width="50%">

### 📋 Categorical Analysis
Value counts table + bar chart for any categorical column. Adjustable Top N slider to control how many categories to show.

</td>
</tr>
<tr>
<td width="50%">

### 🧹 Data Cleaning Pipeline
Missing value imputation (median, mean, mode, constant, or random sampling), duplicate removal with custom strategies, and outlier handling via IQR or Z-score.

</td>
<td width="50%">

### 💾 Export Everything
Download your cleaned dataset as **CSV** or **Excel**, plus a full **JSON cleaning report** that logs every operation applied.

</td>
</tr>
</table>

<br/>

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/AryaBuwa/eda-tool.git
cd eda-tool

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run main.py
```

Open your browser at `http://localhost:8501` — that's it.

---

## Supported Inputs

| Format | Extension |
|--------|-----------|
| CSV | `.csv` |
| Excel | `.xlsx` `.xls` |

---

## Cleaning Options

| Category | Options |
|----------|---------|
| **Missing Values** | Keep / Drop rows / Median+Mode / Mean+Mode / Constant / Random sampling |
| **Duplicates** | Drop all / Keep first / Keep last / Custom column subset |
| **Outliers** | IQR remove / IQR flag / Z-score remove / Z-score flag |
| **Columns** | Drop any column before cleaning |

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| `streamlit` | UI framework |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `plotly` | Interactive charts |
| `openpyxl` | Excel export |

---

<div align="center">

Built by [AryaBuwa](https://github.com/AryaBuwa) &nbsp;·&nbsp; assisted by [Claude AI](https://claude.ai)

</div>