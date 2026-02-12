<div align="center">
<!-- Note: Please create an 'assets' folder and place your project logo or cover image here -->
<img src="https://www.google.com/search?q=https://upload.wikimedia.org/wikipedia/commons/thumb/d/d0/GHN_logo.png/800px-GHN_logo.png" alt="GHN Logo" width="200">

<h1>GHN Executive Dashboard & AI Predictive</h1>

<h3>IBCS Standard · Operations Insights · Flight Risk Radar · Machine Learning</h3>

<p>
<img src="https://www.google.com/search?q=https://img.shields.io/badge/Python-3.10%2B-3776AB%3Fstyle%3Dflat%26logo%3Dpython%26logoColor%3Dwhite" alt="Python">
<img src="https://www.google.com/search?q=https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B%3Fstyle%3Dflat%26logo%3Dstreamlit%26logoColor%3Dwhite" alt="Streamlit">
<img src="https://www.google.com/search?q=https://img.shields.io/badge/Machine%2520Learning-Random%2520Forest-F7931E%3Fstyle%3Dflat%26logo%3Dscikit-learn%26logoColor%3Dwhite" alt="ML">
</p>

</div>

GHN Executive Dashboard is a comprehensive operational and human resources analytics system. Moving beyond traditional static reports, this system provides deep interactivity and data visualization strictly adhering to International Business Communication Standards (IBCS).

AI Integration: Beyond Descriptive Analytics, the system leverages a Random Forest machine learning model to transition into Predictive Analytics, autonomously assessing and flagging employee flight risks before they occur.

<table align="center">
<tr align="center">
<td align="center" valign="top">
<p align="center">
<!-- Replace with actual Dashboard UI screenshot -->
<img src="https://www.google.com/search?q=https://placehold.co/400x240/F5F7F9/FF4D00%3Ftext%3DExecutive%2BDashboard%2BUI" width="400" height="240">




<b>IBCS Standard Interface</b>
</p>
</td>
<td align="center" valign="top">
<p align="center">
<!-- Replace with actual ML Tab screenshot -->
<img src="https://www.google.com/search?q=https://placehold.co/400x240/F5F7F9/FF4D00%3Ftext%3DAI%2BPredictive%2BModel" width="400" height="240">




<b>AI Flight Risk Radar</b>
</p>
</td>
</tr>
</table>

Release Notes

2026-02-12 Launched version 1.0: Finalized Multi-language Dashboard capabilities and successfully integrated the Random Forest machine learning model.

Key Features

IBCS Standard Compliance: Minimalist design with zero visual noise. Strict color coding (Grey for actuals, Red for risks, Green for positive variance) optimizes executive focus.

Machine Learning Integration: Utilizes a Random Forest Classifier to evaluate Feature Importance and predict the probability of resignation (Flight Risk) for individual employees.

Internationalization (I18N): Seamless switching between English and Vietnamese with a single click, catering to diverse management teams and expatriate executives.

Logistics Deep-Dive: Goes beyond standard HR metrics by mining industry-specific insights, such as Late vs. Overtime correlation, Delivery Route Characteristics (Urban vs. Suburban), and Warehouse Health Matrix.

Criteria

Traditional Excel/BI

GHN Executive Dashboard

Interactivity

Static / Limited

Real-time & Highly Interactive

Analytical Perspective

Descriptive (Past)

Predictive (Future)

Design & Color

Subjective, often cluttered

Professional IBCS Standard

Deployment Cost

Requires Licenses (PowerBI/Tableau)

Open-source, Free Deployment

Analytical Workflow

<table align="center">
<tr align="center">
<th><p align="center">Operations Optimization</p></th>
<th><p align="center">Workforce Health</p></th>
<th><p align="center">AI Early Warning</p></th>
</tr>
<tr>
<td align="center"><p align="center"><img src="https://www.google.com/search?q=https://placehold.co/240x180/F5F7F9/FF4D00%3Ftext%3DOps%2BChart" width="240" height="180"></p></td>
<td align="center"><p align="center"><img src="https://www.google.com/search?q=https://placehold.co/240x180/F5F7F9/FF4D00%3Ftext%3DHR%2BChart" width="240" height="180"></p></td>
<td align="center"><p align="center"><img src="https://www.google.com/search?q=https://placehold.co/240x180/F5F7F9/FF4D00%3Ftext%3DML%2BChart" width="240" height="180"></p></td>
</tr>
<tr>
<td align="center">On-time Rate • Overtime • Turn-over</td>
<td align="center">Tenure • eNPS Score • Attrition Trend</td>
<td align="center">Flight Risk • Feature Importance</td>
</tr>
</table>

Installation & Setup

The project is structured to run smoothly on any local environment with Python installed.

1. Clone the repository

git clone [https://github.com/LQP-CTER/GHN-executive-dashboard.git](https://github.com/LQP-CTER/GHN-executive-dashboard.git)
cd GHN-executive-dashboard


2. Install dependencies
(It is recommended to create a virtual environment venv before installation)

pip install -r requirements.txt


3. Run the application

streamlit run app.py


The application will automatically open in your browser at http://localhost:8501.

Data Architecture

The system operates on four primary data sources located in the root directory:

staff_info.csv: Employee demographic details, roles, and assigned warehouses.

employee_logs.csv: Daily operational check-in/check-out logs.

engagement_survey_raw.csv: Raw employee Net Promoter Score (eNPS) survey data.

attrition_data.csv: Historical records of resigned employees and corresponding reasons.

Author

Developed by Le Quy Phat - Data Analyst Candidate.

This project was developed as a technical assessment to demonstrate the practical application of advanced data analytics and machine learning within a Logistics corporate environment.
