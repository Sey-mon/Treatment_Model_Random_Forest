# Child Malnutrition Assessment System

A comprehensive Random Forest-based prescriptive analytics system for assessing malnutrition in children aged 0-5 years, following WHO guidelines and implementing the clinical decision flowchart for treatment recommendations.

## 🎯 Overview

This system provides:
- **Automated WHZ score calculation** based on WHO reference standards
- **Random Forest prediction model** for malnutrition classification
- **Treatment recommendations** following the clinical flowchart provided
- **Interactive web interface** for single and batch assessments
- **Flexible data import/export** capabilities
- **Support for custom WHO charts** for enhanced accuracy

## 📊 Classification Categories

The system classifies children into the following nutritional status categories:

1. **Normal** (WHZ ≥ -2)
2. **Moderate Acute Malnutrition (MAM)** (-3 ≤ WHZ < -2)
3. **Severe Acute Malnutrition (SAM)** (WHZ < -3 or presence of edema)

## 📋 Required Data Fields

### Basic Information
- **name**: Child's name
- **municipality**: City/Municipality
- **number**: Patient/Child ID
- **age_months**: Age in months (0-60)
- **sex**: Male/Female
- **date_of_admission**: Date of assessment

### Household Information
- **total_household**: Total number of household members
- **adults**: Number of adults in household
- **children**: Number of children in household
- **twins**: 0 or 1 (indicating if child is a twin)

### Socio-economic
- **4ps_beneficiary**: Yes/No (Pantawid Pamilyang Pilipino Program beneficiary)

### Anthropometric Measurements
- **weight**: Weight in kilograms
- **height**: Height in centimeters

### Medical History
- **breastfeeding**: Yes/No
- **edema**: True/False
- **tuberculosis**: Yes/No
- **malaria**: Yes/No
- **congenital_anomalies**: Yes/No
- **other_medical_problems**: Yes/No

## 🚀 Quick Start

### Installation

1. **Clone or download the project**
2. **Install Python 3.8 or higher**
3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the web interface:**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

### Using the System

#### Single Patient Assessment
1. Navigate to "Single Patient Assessment"
2. Fill in the patient information form
3. Click "Assess Nutritional Status"
4. Review the results and treatment recommendations

#### Batch Assessment
1. Navigate to "Batch Assessment"
2. Download the CSV template
3. Fill in your patient data
4. Upload the completed CSV file
5. Click "Assess All Patients"
6. Download the results

## 🏗️ System Architecture

```
Treatment_Model_Random_Forest/
├── malnutrition_model.py      # Core model and WHO calculator
├── app.py                     # Streamlit web interface
├── data_manager.py            # Data import/export utilities
├── requirements.txt           # Required Python packages
├── README.md                 # This documentation
└── sample_data/              # Generated sample datasets
```

### Core Components

#### 1. WHO_ZScoreCalculator
- Calculates Weight-for-Height Z-scores
- Implements WHO reference standards
- Supports both boys and girls

#### 2. MalnutritionRandomForestModel
- Random Forest classifier for prediction
- Feature preprocessing and encoding
- Treatment recommendation engine
- Model persistence (save/load)

#### 3. DataManager
- Import/export multiple file formats (CSV, Excel, JSON)
- Data validation and cleaning
- WHO reference data management
- Sample data generation

#### 4. Streamlit Web App
- Interactive patient assessment
- Batch processing capabilities
- Data visualization and analytics
- Results export functionality

## 🔬 Model Details

### Features Used
- **Anthropometric**: age_months, weight, height, BMI, WHZ_score
- **Demographic**: sex, age_group
- **Household**: total_household, adults, children, twins
- **Socio-economic**: 4ps_beneficiary, municipality
- **Medical**: breastfeeding, tuberculosis, malaria, congenital_anomalies, other_medical_problems

### Model Performance
- **Algorithm**: Random Forest Classifier
- **Cross-validation**: 5-fold CV
- **Features**: Automatic importance ranking
- **Validation**: Built-in train/test split

## 🏥 Treatment Recommendations

The system follows the clinical decision flowchart and provides specific treatment recommendations:

### Severe Acute Malnutrition (SAM)
- **With edema**: Inpatient therapeutic care with stabilization protocols
- **Without edema**: Outpatient therapeutic care with RUTF (Ready-to-Use Therapeutic Food)

### Moderate Acute Malnutrition (MAM)
- Targeted supplementary feeding program
- 75 kcal/kg/day supplementation
- Regular monitoring and follow-up

### Normal
- Routine health check and nutrition care
- Preventive interventions
- Regular growth monitoring

## 📁 Data Import/Export

### Supported Formats
- **CSV**: Comma-separated values
- **Excel**: .xlsx format
- **JSON**: JavaScript Object Notation

### Data Validation
- Automatic data type conversion
- Range validation for measurements
- Standardization of categorical values
- Missing data handling

### Sample Data Template
Download the template from the web interface or generate programmatically:

```python
from data_manager import DataManager
dm = DataManager()
template = dm.create_data_template('template.xlsx')
```

## 🎛️ Customization

### Adding Custom WHO Charts
You can import your own WHO reference charts:

```python
from data_manager import DataManager
dm = DataManager()
dm.load_who_charts('custom_who_charts.xlsx')
```

### Model Retraining
Retrain the model with new data:

```python
from malnutrition_model import MalnutritionRandomForestModel
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Initialize and train model
model = MalnutritionRandomForestModel()
model.train_model(df)
model.save_model('updated_model.pkl')
```

## 📊 Example Usage

### Python API
```python
from malnutrition_model import MalnutritionRandomForestModel

# Initialize model
model = MalnutritionRandomForestModel()

# Single prediction
patient_data = {
    'name': 'Test Child',
    'age_months': 18,
    'sex': 'Male',
    'weight': 8.5,
    'height': 76.0,
    'municipality': 'Manila',
    '4ps_beneficiary': 'Yes',
    # ... other fields
}

result = model.predict_single(patient_data)
print(f"Status: {result['prediction']}")
print(f"WHZ Score: {result['whz_score']}")
print(f"Treatment: {result['recommendation']['treatment']}")
```

## 🚨 Important Notes

### Data Quality
- Ensure accurate measurements (weight, height)
- Verify age in months (0-60 range)
- Complete all required fields for best results

### Clinical Use
- This system is designed as a **decision support tool**
- **Always consult qualified healthcare professionals**
- Use in conjunction with clinical assessment
- Regular model validation recommended

### WHO Guidelines Compliance
- Follows WHO Child Growth Standards
- Implements standard Z-score calculations
- Adheres to malnutrition classification criteria

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**
   - Check file format and column names
   - Ensure data is within valid ranges
   - Review sample template for format

2. **Model Performance**
   - Increase training data size
   - Check for data quality issues
   - Validate feature importance

3. **Web Interface Issues**
   - Restart Streamlit application
   - Check browser compatibility
   - Clear browser cache

### Getting Help
1. Check the console output for error messages
2. Validate your data against the template
3. Ensure all required packages are installed
4. Review the sample data format

## 📈 Future Enhancements

### Planned Features
- [ ] Integration with full WHO reference tables
- [ ] Multi-language support
- [ ] Advanced reporting and analytics
- [ ] Integration with health information systems
- [ ] Mobile-responsive interface
- [ ] Automated alert systems
- [ ] Longitudinal tracking capabilities

### Contributing
This system is designed for educational and research purposes. For production use in healthcare settings, additional validation and clinical testing would be required.

## 📄 License

This project is intended for educational and research purposes. Please ensure compliance with local healthcare regulations when adapting for clinical use.

## 🙏 Acknowledgments

- WHO Child Growth Standards and Guidelines
- Scikit-learn machine learning library
- Streamlit for web interface
- Plotly for interactive visualizations

---

**Note**: This system implements the clinical decision flowchart provided and follows WHO guidelines for child malnutrition assessment. It should be used as a decision support tool in conjunction with professional clinical judgment.
