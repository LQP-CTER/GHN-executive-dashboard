GHN Executive Dashboard & AI Predictive Analytics

Tech Stack: Python | Streamlit | Machine Learning (Random Forest) | Plotly

Project Overview

The GHN Executive Dashboard is an interactive web application designed specifically for the Operations and HR Executive Board.

Unlike traditional static reports, this system not only provides a holistic view of operational performance (Descriptive Analytics) but also integrates Machine Learning models to forecast employee turnover risks (Predictive Analytics), thereby enabling the business to make timely interventions.

🔗 Direct Access Link: [[GHN-executive-dashboard.streamlit.app](https://ghn-executive-dashboard.streamlit.app/)]

Key Features

IBCS Standard Interface: Minimalist design adhering to International Business Communication Standards. Strict color coding rules (Grey for actuals, Red for risks, Green for positive variance) help viewers focus 100% on the data.

Machine Learning Integration: Applies the Random Forest Classifier algorithm to evaluate "Turnover Drivers" and provides a "Flight Risk Radar" for all active personnel.

Multi-language (I18N): Supports seamless switching between Vietnamese and English (VN/EN) directly on the sidebar.

Logistics Industry Insights: Simulates and analyzes specific delivery industry metrics such as: Delivery Route Type (Urban Apartments vs. Suburban), Shipper Classification (Hybrid vs. Dedicated), and Warehouse Health Matrix.

Dashboard Architecture

The system is divided into 5 main modules:

Executive: Core KPI cards (SLA, On-time Rate, Turnover Rate, Happiness Score) and overall trends.

Operations: Deep dive into the causes of late arrivals, overwork risks, and the Warehouse/Hub performance warning matrix.

HR & Attrition: Correlation analysis between tenure, work environment, and turnover rate.

Data Quality: Transparency check comparing Raw data with Standardized data.

ML Predict: Feature Importance charts and a list of high-risk employees requiring urgent 1-on-1 meetings for retention.

Local Setup

Step 1: Clone the repository

git clone [https://github.com/LQP-CTER/GHN-executive-dashboard.git](https://github.com/LQP-CTER/GHN-executive-dashboard.git)
cd GHN-executive-dashboard


Step 2: Install required libraries
(It is recommended to create a virtual environment venv before installation)

pip install -r requirements.txt


Step 3: Run the application

streamlit run app.py


The application will automatically open in your browser at http://localhost:8501.

Folder Structure

GHN-executive-dashboard/
│
├── app.py                      # Main Streamlit Dashboard code
├── requirements.txt            # Required Python libraries list
├── README.md                   # Project documentation
│
├── attrition_data.csv          # Employee resignation data
├── employee_logs.csv           # Historical check-in/check-out logs
├── engagement_survey_raw.csv   # Raw employee satisfaction survey data
└── staff_info.csv              # Employee list and demographics


Developed by [Le Quy Phat] - Operations Data Analyst Candidate.
