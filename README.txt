# Social Media Fraud Detection - ML Pipeline

## 📋 Project Overview

This project is a comprehensive machine learning pipeline designed to detect fake/fraudulent social media accounts using advanced ensemble methods and classification algorithms. The system analyzes user profile characteristics and behavioral patterns to identify potentially fraudulent accounts with high accuracy.

## 🎯 What This Project Does

Our fraud detection system evaluates social media accounts based on multiple features including:
- Profile completeness (profile picture, bio)
- Account activity metrics (followers, following count)
- Username characteristics and randomness
- Behavioral patterns and engagement metrics

The system provides:
- Automated training pipeline with multiple ML models
- Model comparison and performance evaluation
- Interactive web application for real-time predictions
- Comprehensive visualizations and analytics

## 🛠️ How We Built It

### Technologies & Libraries Used

**Core ML & Data Science:**
- Python 3.x
- Scikit-learn - Model training and evaluation
- XGBoost - Gradient boosting implementation
- Pandas - Data manipulation and analysis
- NumPy - Numerical computations
- Joblib - Model serialization

**Visualization:**
- Matplotlib - Static plots and visualizations
- Seaborn - Statistical data visualization

**Web Application:**
- Streamlit - Interactive web interface for predictions

### Machine Learning Pipeline

1. **Data Preprocessing:**
   - Feature engineering and selection
   - Missing value handling
   - Feature scaling and normalization
   - Train-test split (80-20)

2. **Model Training:**
   We trained and compared 4 different classification algorithms:
   - Gradient Boosting Classifier (Best Performer)
   - Random Forest Classifier
   - XGBoost Classifier
   - Logistic Regression

3. **Model Evaluation:**
   - ROC-AUC scores for all models
   - Precision, Recall, F1-Score metrics
   - Confusion matrices
   - ROC curves visualization

4. **Model Selection:**
   - Automated selection of best performing model
   - Cross-validation for robust performance estimation

## 📁 Project Structure

```
├── fake_dataset.csv              # Training dataset
├── fake_dataset.xlsx             # Dataset (Excel format)
├── fakeaccount_pipeline.ipynb    # Jupyter notebook with full pipeline
├── fakeaccount_pipeline.py       # Python script version of pipeline
├── inspect_model.py              # Model inspection and analysis tool
├── streamlit_manual_predict.py   # Web app for predictions
├── requirements.txt              # Project dependencies
├── README.txt                    # This file
└── outputs/
    ├── best_model_GradientBoosting.joblib       # Best trained model
    ├── pipeline_*.joblib                        # All trained pipelines
    ├── models_summary_metrics.csv               # Performance metrics
    ├── models_auc_summary.csv                   # AUC scores
    ├── roc_curves.png                           # ROC curves comparison
    ├── sample_preds_*.csv                       # Sample predictions
    └── plots/                                   # EDA visualizations
        ├── class_counts.png
        ├── corr_heatmap.png
        ├── dist_*.png                           # Feature distributions
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Required Packages
The project uses the following key packages:
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- streamlit
- joblib

## 💻 How to Use

### Option 1: Run the Full Pipeline

**Using Jupyter Notebook:**
```bash
jupyter notebook fakeaccount_pipeline.ipynb
```
Run all cells to execute the complete pipeline from data loading to model training.

**Using Python Script:**
```bash
python fakeaccount_pipeline.py
```
This will automatically:
- Load and preprocess data
- Train all models
- Generate evaluation metrics
- Save models and visualizations to ./outputs/

### Option 2: Interactive Web Application

Launch the Streamlit web app for real-time predictions:
```bash
streamlit run streamlit_manual_predict.py
```

Features:
- Manual input of account features
- Real-time fraud prediction
- Probability scores for each class
- User-friendly interface

### Option 3: Model Inspection

Analyze trained models in detail:
```bash
python inspect_model.py
```

This provides:
- Feature importance analysis
- Model statistics
- Detailed performance metrics

## 📊 Model Performance

Our best performing model (Gradient Boosting) achieves:
- High accuracy in detecting fraudulent accounts
- Excellent ROC-AUC scores
- Balanced precision and recall
- Robust performance on unseen data

All performance metrics are saved in:
- `outputs/models_summary_metrics.csv`
- `outputs/models_auc_summary.csv`

## 🔍 Features Analyzed

The model considers multiple features including:
1. **Profile Features:**
   - Has profile picture
   - Bio/description length
   - Name characteristics

2. **Network Features:**
   - Follower count
   - Following count
   - Follower/following ratio

3. **Account Characteristics:**
   - Username randomness score
   - Account age indicators
   - Activity patterns

## 📈 Output Files

After running the pipeline, you'll find:

- **Models:** Trained and serialized model files (.joblib)
- **Metrics:** CSV files with detailed performance metrics
- **Visualizations:** 
  - ROC curves for all models
  - Feature distributions
  - Correlation heatmaps
  - Class balance plots
- **Predictions:** Sample predictions with probabilities

## 🔮 Future Enhancements

Potential improvements for this project:
- Deep learning models (Neural Networks)
- Real-time data integration with social media APIs
- Additional behavioral features
- Ensemble stacking techniques
- Model deployment to cloud platforms
- API for batch predictions

## 🤝 Contributing

This project was built as a machine learning proof-of-concept for detecting fraudulent social media accounts using supervised learning techniques.

## 📝 Notes

- The dataset contains synthetic/sample data for demonstration purposes
- Models are pre-trained and saved in the outputs folder
- All visualizations are automatically generated during pipeline execution
- The pipeline supports easy addition of new models

## 🎓 Learning Outcomes

This project demonstrates:
- End-to-end ML pipeline development
- Multiple model comparison and selection
- Feature engineering for fraud detection
- Model evaluation and validation techniques
- Production-ready ML application development

---

For questions or issues, please refer to the code documentation or open an issue on GitHub.
