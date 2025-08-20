# Parkinson's Disease Predictor with XAI 🧠🔍

An advanced machine learning project for predicting Parkinson's Disease using voice measurements, enhanced with Explainable AI (XAI) techniques like LIME and SHAP for transparent and interpretable results.

## ✨ Features

- **Machine Learning Models**: Implements Random Forest and SVM classifiers for accurate Parkinson's prediction.
- **Data Handling**: Loads and preprocesses the UCI Parkinson's dataset, handling missing values and feature splitting.
- **Model Evaluation**: Includes accuracy scores, confusion matrices, and classification reports.
- **Explainable AI (XAI)**: 
  - Uses **LIME** (Local Interpretable Model-agnostic Explanations) to generate instance-level explanations, visualizing feature contributions.
  - Integrates **SHAP** (SHapley Additive exPlanations) for global and local interpretability, highlighting how features like voice jitter and shimmer influence predictions.
- **Visualization**: Confusion matrix plots and LIME explanations displayed in Jupyter Notebook.
- **Reproducible**: Fixed random states for consistent results.

## 🎯 Why XAI Matters Here
Parkinson's prediction relies on subtle voice features (e.g., MDVP:Jitter, PPE). Traditional ML models are black boxes, but with LIME and SHAP, we make predictions transparent—doctors can trust and understand why a patient is classified as having Parkinson's, improving clinical adoption.

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or Lab
- The UCI Parkinson's dataset (`parkinsons.data`) – download from [UCI ML Repository](https://archive.ics.uci.edu/dataset/174/parkinsons) and place in the project root.

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/parkinsons-predictor-xai.git
   cd parkinsons-predictor-xai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add the Dataset**
   - Download `parkinsons.data` from the UCI link above.
   - Update the file path in the notebook if needed (default: reads from root).

4. **Run the Notebook**
   ```bash
   jupyter notebook Parkinson_Disease_Prediction.ipynb
   ```

## 🔧 Configuration

- No API keys required—just ensure the dataset is available.
- For SHAP integration: Add SHAP explanations by installing `shap` (already in requirements) and adding code like:
  ```python
  import shap
  explainer = shap.Explainer(rf_model)
  shap_values = explainer(X_test)
  shap.summary_plot(shap_values, X_test)
  ```
- Customize model hyperparameters in the notebook (e.g., random_state=45).

## 📖 Usage

### Running the Model
1. Open `Parkinson_Disease_Prediction.ipynb` in Jupyter.
2. Execute cells sequentially:
   - Load and preprocess data.
   - Train/test split.
   - Train Random Forest/SVM.
   - Evaluate models.
   - Generate LIME explanations for specific instances.
3. For SHAP: Add the snippet above to visualize feature importances.

### Example Output
- Accuracy: ~95% on test set.
- LIME Explanation (for a sample instance):
  ![LIME Example](assets/lime-example.png) *(Add screenshots to /assets folder)*
- SHAP Summary Plot: Shows global feature impacts, e.g., high PPE values strongly predict Parkinson's.

### Predicting New Data
- Prepare a DataFrame with features (e.g., MDVP:Fo(Hz), Jitter, etc.).
- Use `rf_model.predict(new_data)` for predictions.
- Explain with LIME: `explainer.explain_instance(new_instance, rf_model.predict_proba)`.

## 📊 Dataset Overview
- Source: UCI Machine Learning Repository.
- Features: 23 attributes from voice recordings (e.g., frequency measures, jitter, shimmer).
- Target: 'status' (0: Healthy, 1: Parkinson's).
- Samples: 195 instances.

## 📁 Project Structure

```
parkinsons-predictor-xai/
├── Parkinson_Disease_Prediction.ipynb  # Main Jupyter Notebook (copy-paste your code here)
├── parkinsons.data                     # Dataset (download and add)
├── requirements.txt                    # Dependencies
├── README.md                           # This file
├── .gitignore                          # Git ignore rules
├── LICENSE                             # MIT License
└── assets/                             # For screenshots/images (optional)
```

## 🛡️ Security Notes
- No external APIs or sensitive data involved.
- Ensure dataset is handled ethically (anonymized voice data).

## 🚨 Troubleshooting

**Dataset Not Found**
- Verify `parkinsons.data` is in the root and update the path in the notebook.

**LIME/SHAP Errors**
- Ensure `lime` and `shap` are installed.
- For visualization issues: Run in Jupyter (not VS Code) or install `ipywidgets`.

**Low Accuracy**
- Experiment with hyperparameters or add more models (e.g., XGBoost).

## 🔄 Updates
- v1.0: Initial release with RF/SVM and LIME.
- Future: Add SHAP dashboards, web app deployment.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing
1. Fork the repository.
2. Create a feature branch.
3. Commit changes (e.g., add SHAP code).
4. Push and open a Pull Request.

## 📞 Support
- Open an issue on GitHub for bugs/questions.
- Inspired by UCI datasets and XAI libraries.

## ⚠️ Disclaimer
This is for educational/research purposes. Not a medical diagnostic tool—consult professionals for health advice.

---

**Built with curiosity by Ansh Chhibber**
