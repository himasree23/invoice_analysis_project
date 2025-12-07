# 📊 Invoice Analysis & Prediction System

A complete end-to-end machine learning project for invoice data extraction, analysis, and prediction.

## 🎯 Features

- **PDF Invoice Extraction**: Automatically extract data from PDF invoices
- **Data Processing**: Clean and transform invoice data
- **Machine Learning**: Train predictive models using XGBoost, Random Forest, and LightGBM
- **Web Interface**: Beautiful Streamlit dashboard for easy interaction
- **Analytics**: Visualize spending patterns and trends
- **Predictions**: Forecast future invoice amounts
- **Cloud Deployment**: Deploy to Streamlit Cloud, Heroku, or other platforms

## 📋 Prerequisites

- **Python**: 3.10.x or 3.11.x
- **VS Code**: Latest version
- **Operating System**: Windows 10/11

## 🚀 Quick Start (Windows)

### Step 1: Clone or Download Project

```bash
mkdir invoice_analysis_project
cd invoice_analysis_project
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Command Prompt)
venv\Scripts\activate

# Activate (PowerShell)
venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
invoice_analysis_project/
│
├── venv/                      # Virtual environment
├── data/
│   ├── raw/                   # PDF invoices
│   ├── processed/             # Processed CSV files
│   └── sample_invoices.csv    # Sample data
│
├── models/                    # Saved ML models
│
├── src/
│   ├── __init__.py
│   ├── pdf_extractor.py      # PDF extraction
│   ├── data_processor.py     # Data cleaning
│   ├── model_builder.py      # ML models
│   └── analyzer.py           # Analysis tools
│
├── app.py                    # Streamlit web app
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── .gitignore              # Git ignore
```

## 🎮 Usage

### Option 1: Use the Web Interface (Recommended)

1. Run `streamlit run app.py`
2. Upload your data or generate sample data
3. Explore data in the overview page
4. Train a model in the training page
5. Make predictions and view analytics

### Option 2: Use Command Line

```bash
# Extract PDF invoices
python src/pdf_extractor.py

# Process data
python src/data_processor.py

# Train model
python src/model_builder.py
```

## 📦 Dependencies

Core libraries:
- **pandas** (2.1.1) - Data manipulation
- **numpy** (1.26.0) - Numerical computing
- **scikit-learn** (1.3.1) - Machine learning
- **xgboost** (2.0.0) - Gradient boosting
- **streamlit** (1.27.2) - Web application
- **plotly** (5.17.0) - Interactive visualizations

See `requirements.txt` for complete list.

## 🤖 Machine Learning Models

The system supports multiple ML algorithms:
- **XGBoost** (Recommended)
- **Random Forest**
- **LightGBM**
- **Gradient Boosting**
- **Linear Regression**

## 📊 Data Format

Expected CSV format:
```csv
invoice_number,date,vendor,amount,tax,filename
INV-1001,2024-01-15,ABC Corp,1250.00,125.00,invoice_1.pdf
```

## 🌐 Deployment

### Deploy to Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy!

### Deploy to Heroku

```bash
# Create Procfile
echo "web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

## 🐛 Troubleshooting

### "python is not recognized"
- Reinstall Python and check "Add to PATH"

### "Cannot activate virtual environment"
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### "Module not found"
```bash
pip install --upgrade -r requirements.txt
```

### Streamlit won't open
```bash
streamlit run app.py --server.port 8502
```

## 📈 Performance

Typical model performance:
- **R² Score**: 0.85-0.95
- **RMSE**: $50-$150 (depending on data)
- **Training Time**: 1-3 minutes

## 🔧 Customization

### Add New Features

Edit `src/data_processor.py` in the `feature_engineering()` method:

```python
def feature_engineering(self):
    # Add your custom features here
    self.df['custom_feature'] = self.df['amount'] * 2
    return self.df
```

### Change Model Parameters

Edit `src/model_builder.py` in the `train_model()` method:

```python
self.model = xgb.XGBRegressor(
    n_estimators=200,  # Change this
    learning_rate=0.05,  # And this
    max_depth=7
)
```

## 📝 License

MIT License - Feel free to use for personal or commercial projects

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📧 Support

For issues or questions:
- Check the troubleshooting section
- Review error messages in the console
- Ensure all dependencies are installed

## 🎓 Learning Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Scikit-learn Guide](https://scikit-learn.org/stable/user_guide.html)
- [XGBoost Tutorials](https://xgboost.readthedocs.io)

## 🚧 Future Enhancements

- [ ] Multi-currency support
- [ ] Email notifications
- [ ] Automated report generation
- [ ] Database integration
- [ ] User authentication
- [ ] Real-time data updates
- [ ] Advanced anomaly detection
- [ ] Mobile app

## ⭐ Star History

If you find this project useful, please give it a star!

---

Made with ❤️ by Your Team

Last Updated: December 2024