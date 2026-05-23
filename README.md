# Iris Species Classification Project

## Team Members
- Jose Elias Mendez Rizcala
- Alban David Amaya Lobelo
- Natalia Mendoza
- Loraine De Castro

## Project Purpose
This project implements a complete data mining pipeline for classifying Iris flowers into three species (**Setosa**, **Versicolor**, **Virginica**) based on four numerical features: sepal length, sepal width, petal length, and petal width.

The workflow includes:
- Data understanding and preprocessing
- Training a **Random Forest** classifier
- Model evaluation (accuracy, precision, recall, F1-score)
- An interactive **Streamlit dashboard** for real-time predictions and visualizations

## Repository Contents
| File | Description |
|------|-------------|
| `Project.py` | Streamlit dashboard application |
| `requirements.txt` | Python dependencies |
| `Iris.csv` | Dataset (150 samples) |
| `iris_rf_model.pkl` | Trained Random Forest model |
| `scaler.pkl` | StandardScaler for feature normalization |
| `label_encoder.pkl` | Label encoder for species names |
| `README.md` | This file |

## How to Run the Dashboard Locally

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd <repo-folder>