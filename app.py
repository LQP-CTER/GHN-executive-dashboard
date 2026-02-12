import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

# config trang web
st.set_page_config(
    page_title="GHN Executive Dashboard",
    page_icon="GHN",
    layout="wide",
    initial_sidebar_state="expanded"
)

# tu dien de chuyen doi ngon ngu
TRANSLATIONS = {
    'VN': {
        'app_title': "GHN Executive Dashboard",
        'app_sub': "*Báo cáo phân tích chuyên sâu tuân thủ Tiêu chuẩn Giao tiếp Kinh doanh Quốc tế (IBCS)*",
        'tab_exec': "Tổng quan", 'tab_ops': "Vận hành", 'tab_hr': "Nhân sự", 'tab_data': "Dữ liệu", 'tab_ml': "Machine Learning",
        'filter_title': "BỘ ĐIỀU KHIỂN", 'filter_wh': "Chọn Kho", 'filter_role': "Vị trí", 'filter_date': "Khoảng thời gian",
        'all_wh': "Chọn tất cả Kho", 'all_role': "Chọn tất cả Vị trí",
        
        # may cai label cho kpi
        'sla_title': "CHỈ SỐ CAM KẾT DỊCH VỤ (SLA & OPERATIONS)",
        'hr_title': "CHỈ SỐ NGUỒN NHÂN LỰC (HR & WORKFORCE)",
        'ops_kpi_title': "CHỈ SỐ HIỆU SUẤT VẬN HÀNH (OPERATIONS KPIs)",
        'hr_kpi_title': "CHỈ SỐ HIỆU SUẤT NHÂN SỰ (HR KPIs)",
        'sla_del': "SLA Giao hàng toàn trình (Demo)", 'sla_wh': "SLA Xử lý tại Kho (Demo)",
        'sla_att': "SLA Tuân thủ Giờ làm", 'sla_overload': "Tỷ lệ Ca làm quá tải (>10h)",
        'kpi_staff': "Tổng nhân sự (Active)", 'kpi_ontime': "Tỷ lệ đi làm đúng giờ", 
        'kpi_resign': "Số lượng nghỉ việc", 'kpi_happy': "Chỉ số Hạnh phúc (TB)",
        'kpi_unit_p': "Người", 'kpi_alert': "Báo động đỏ", 'kpi_avg': "Mức trung bình",
        'late_count': "Tổng lượt đi muộn", 'avg_hr': "Giờ làm TB / Ngày", 'turnover_rate': "Tỷ lệ Turnover",
        
        # ten chart
        'c1_title': "Tỷ trọng Tuân thủ quy định giờ làm việc", 'c1_sub': "Cơ cấu tỷ lệ phần trăm các lượt Check-in Đúng giờ so với Đi muộn",
        'c2_title': "WH01 và WH05 là 2 kho vi phạm mục tiêu đúng giờ", 'c2_sub': "Tỷ lệ On-time (%) phân tách theo từng Kho",
        'c3_title': "Số lượng nhân sự rời đi duy trì ở mức cao", 'c3_sub': "Tổng số lượng nhân sự nghỉ việc theo các tháng gần nhất",
        'c4_title': "Vấn đề Quản lý (Management) là nguyên nhân gốc rễ", 'c4_sub': "Phân bổ số lượng nhân sự nghỉ việc theo lý do",
        'c5_title': "Thứ 2 và Chủ Nhật có tỷ lệ đi muộn cao nhất tuần", 'c5_sub': "Phân tích 'Monday Blues': Tỷ lệ đi muộn (%) theo thứ trong tuần",
        'c6_title': "Dấu hiệu kiệt sức: Làm thêm giờ dẫn tới đi muộn", 'c6_sub': "Tương quan giữa Giờ làm trung bình và Tổng số lần đi muộn / Nhân sự",
        'c7_title': "Shipper có xu hướng làm việc quá giờ chuẩn (8h)", 'c7_sub': "Số giờ làm việc trung bình phân tách theo Vị trí (Role)",
        'c8_title': "Cảnh báo vi phạm chính sách Vận hành", 'c8_sub': "Danh sách nhân sự làm việc vượt quá 16 tiếng/ca (Rủi ro an toàn)",
        'c9_title': "Phòng Operations có chỉ số Hạnh phúc thấp nhất", 'c9_sub': "Chỉ số Hạnh phúc trung bình (Thang 5) theo Khối/Phòng ban",
        'c10_title': "Nhân sự đánh giá kém về môi trường hơn công việc", 'c10_sub': "So sánh điểm thành phần khảo sát trung bình toàn công ty",
        'c11_title': "Nguy cơ chảy máu nhân sự ở nhóm từ 12-24 tháng", 'c11_sub': "Phân bổ thâm niên (số tháng) của nhóm nhân sự đã nghỉ việc",
        'c12_title': "Warehouse Staff chiếm đa số trong tỷ lệ nghỉ việc", 'c12_sub': "Tổng số lượng nhân sự nghỉ việc phân theo Vị trí (Role)",
        
        # chart insight
        'c13_title': "Áp lực vận hành đè nặng lên nhóm Shipper Kiêm nhiệm", 'c13_sub': "So sánh Giờ làm trung bình theo Loại hình Driver (Mô phỏng)",
        'c14_title': "Ma trận Sức khỏe Bưu cục (Kho)", 'c14_sub': "Theo dõi Tỷ lệ Hoàn hàng vs SLA Giao hàng vs Tỷ lệ Nghỉ việc (Mô phỏng)",
        'c15_title': "Tuyến Nội thành (Chung cư) có Tỷ lệ nghỉ việc cao nhất", 'c15_sub': "Phân tích Tỷ lệ Turnover theo Đặc thù Tuyến giao hàng (Mô phỏng)",
        
        # title model ml
        'ml_title': "PHÂN TÍCH DỰ ĐOÁN NGHỈ VIỆC (MACHINE LEARNING)",
        'c16_title': "Mô hình Rừng ngẫu nhiên: Yếu tố thúc đẩy nghỉ việc", 'c16_sub': " Đánh giá mức độ ảnh hưởng của các chỉ số đến việc nghỉ việc",
        'c17_title': "Radar Cảnh báo: Nhân sự có Rủi ro nghỉ việc cao (Flight Risk)", 'c17_sub': "Dự đoán xác suất % nghỉ việc của nhân sự Active dựa trên dữ liệu hiện tại",
        'ml_algo': "Thuật toán sử dụng: Random Forest Classifier",
        
        # text cho ml
        'ml_algo_desc_title': "🧠 Thuật toán này hoạt động như thế nào?",
        'ml_algo_desc': "Random Forest là thuật toán hoạt động như một hội đồng chuyên gia: nhiều cây quyết định cùng phân tích dữ liệu ngẫu nhiên và bỏ phiếu, kết quả đa số sẽ là dự đoán cuối cùng.",
        'ml_insight': "💡 **Insight:** Các yếu tố nằm ở trên cùng là nguyên nhân chính dẫn đến quyết định nghỉ việc của nhân viên theo mô hình AI phân tích.",
        'ml_action': "⚠️ **Hành động đề xuất (Actionable):** HR/Ops Manager cần có buổi 1-1 (1-on-1 meeting) khẩn cấp với các nhân sự có nguy cơ > 60% ở bảng trên để tìm hiểu tâm tư và giữ chân nhân tài.",
        'ml_no_data': "Không có dữ liệu nhân sự Active để dự đoán.",

        'no_anomaly': "Không phát hiện ca làm việc nào vượt quá 16 tiếng.",
        'x_day': "Ngày", 'y_rate': "Tỷ lệ (%)", 'x_workhr': "Trung bình giờ làm/Ngày", 'y_late': "Số lần đi muộn",
        'status_ontime': "Đúng giờ", 'status_late': "Đi muộn"
    },
    'EN': {
        'app_title': "GHN Executive Dashboard",
        'app_sub': "*In-depth analysis report complying with International Business Communication Standards (IBCS)*",
        'tab_exec': "Overview", 'tab_ops': "Operations", 'tab_hr': "Attrition & HR", 'tab_data': "Data Quality", 'tab_ml': "Machine Learning",
        'filter_title': "CONTROL PANEL", 'filter_wh': "Warehouse", 'filter_role': "Role", 'filter_date': "Period",
        'all_wh': "Select All Warehouses", 'all_role': "Select All Roles",
        
        'sla_title': "SERVICE LEVEL AGREEMENTS (SLA & OPERATIONS)",
        'hr_title': "WORKFORCE & HR METRICS",
        'ops_kpi_title': "OPERATIONS KPIs",
        'hr_kpi_title': "HR & ATTRITION KPIs",
        'sla_del': "E2E Delivery SLA (Demo)", 'sla_wh': "Warehouse Processing SLA (Demo)",
        'sla_att': "Working Time Compliance SLA", 'sla_overload': "Overload Shift Rate (>10h)",
        'kpi_staff': "Total Active Staff", 'kpi_ontime': "On-time Arrival Rate", 
        'kpi_resign': "Total Resignations", 'kpi_happy': "Avg Happiness Score",
        'kpi_unit_p': "Staff", 'kpi_alert': "Red Alert", 'kpi_avg': "Average Level",
        'late_count': "Total Late Check-ins", 'avg_hr': "Avg Work Hours/Day", 'turnover_rate': "Turnover Rate",
        
        'c1_title': "Working Time Compliance Distribution", 'c1_sub': "Percentage breakdown of On-time vs Late check-ins",
        'c2_title': "WH01 and WH05 failed the on-time target", 'c2_sub': "On-time rate (%) breakdown by Warehouse",
        'c3_title': "Resignation numbers remain consistently high", 'c3_sub': "Total resignations by recent months",
        'c4_title': "Management issues are the root cause", 'c4_sub': "Distribution of resignations by reason",
        'c5_title': "Mondays and Sundays see the highest late rates", 'c5_sub': "'Monday Blues' Analysis: Late rate (%) by weekday",
        'c6_title': "Burnout signs: Overtime leads to lateness", 'c6_sub': "Correlation between Avg Work Hours and Total Latenesses / Staff",
        'c7_title': "Shippers tend to work beyond standard hours (8h)", 'c7_sub': "Average work hours breakdown by Role",
        'c8_title': "Operational Policy Violation Alerts", 'c8_sub': "List of staff working > 16 hours/shift (Safety Risk)",
        'c9_title': "Operations department has the lowest Happiness score", 'c9_sub': "Average Happiness Index (out of 5) by Department",
        'c10_title': "Environment rated lower than Work content", 'c10_sub': "Comparison of average survey components company-wide",
        'c11_title': "High flight risk in the 12-24 months tenure group", 'c11_sub': "Tenure distribution (months) of resigned staff",
        'c12_title': "Warehouse Staff dominates the resignation rate", 'c12_sub': "Total resignations distributed by Role",
        
        'c13_title': "Hybrid Shippers bear the highest operational pressure", 'c13_sub': "Average Work Hours compared by Driver Type (Mock)",
        'c14_title': "Warehouse Health Matrix", 'c14_sub': "Monitoring Return Rate vs Delivery SLA vs Turnover (Mock)",
        'c15_title': "Urban routes (Apartments) suffer highest turnover", 'c15_sub': "Turnover Rate analysis by Delivery Zone Characteristic (Mock)",
        
        'ml_title': "ATTRITION PREDICTIVE ANALYTICS (MACHINE LEARNING)",
        'c16_title': "Random Forest: Turnover Drivers", 'c16_sub': "Assessing the importance of factors influencing resignation decisions.",
        'c17_title': "Flight Risk Radar: High Risk Employees", 'c17_sub': "Predicted probability (%) of Active staff resigning based on current data",
        'ml_algo': "Algorithm applied: Random Forest Classifier",

        'ml_algo_desc_title': "🧠 How does this algorithm work?",
        'ml_algo_desc': "Random Forest operates like a council of experts: multiple decision trees analyze random data and vote, with the majority result becoming the final prediction.",
        'ml_insight': "💡 **Insight:** The features at the top are the primary drivers leading to an employee's decision to resign, according to the AI model.",
        'ml_action': "⚠️ **Actionable:** HR/Ops Managers should conduct urgent 1-on-1 meetings with staff showing > 60% flight risk in the table above to understand their concerns and retain talent.",
        'ml_no_data': "No Active staff data available for prediction.",

        'no_anomaly': "No shifts > 16 hours detected.",
        'x_day': "Date", 'y_rate': "Rate (%)", 'x_workhr': "Avg Work Hours/Day", 'y_late': "Times Late",
        'status_ontime': "On-time", 'status_late': "Late"
    }
}

# check xem co session language chua
if 'lang' not in st.session_state:
    st.session_state.lang = 'VN'

# ham dich text
def _(key):
    return TRANSLATIONS[st.session_state.lang].get(key, key)

# set may cai mau de dung lai cho de
IBCS_ACTUAL = "#404040"
IBCS_GOOD = "#92D050"
IBCS_BAD = "#C00000"
IBCS_TARGET = "#000000"
BG_COLOR = "#F5F7F9"
CARD_COLOR = "#FFFFFF"

# custom css cho app dep hon
st.markdown(f"""
<style>
    .stApp {{ background-color: {BG_COLOR}; font-family: 'Segoe UI', sans-serif; }}
    section[data-testid="stSidebar"] {{ background-color: #FFFFFF; border-right: 1px solid #E0E0E0; }}
    .css-card {{ background-color: {CARD_COLOR}; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); margin-bottom: 20px; border: 1px solid #EBEBEB; }}
    div[data-testid="stMetric"] {{ background-color: #FFFFFF; padding: 15px; border-radius: 10px; border: 1px solid #F0F0F0; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }}
    div[data-testid="stMetricLabel"] {{ font-size: 14px; color: #666; font-weight: 600; text-transform: uppercase; }}
    div[data-testid="stMetricValue"] {{ font-size: 26px; color: #333; font-weight: 700; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; background-color: transparent; }}
    .stTabs [data-baseweb="tab"] {{ height: 45px; background-color: #FFFFFF; border-radius: 8px 8px 0 0; padding: 0 20px; font-weight: 600; border: 1px solid #E0E0E0; border-bottom: none; color: #555; }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{ background-color: #FFFFFF; border-top: 3px solid #FF4D00; color: #FF4D00; }}
    .chart-container {{ background-color: #FFFFFF; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); border: 1px solid #E5E7EB; margin-bottom: 20px; height: 100%; }}
    .ibcs-title {{ font-size: 15px; font-weight: 700; color: #111827; margin-bottom: 5px; }}
    .ibcs-subtitle {{ font-size: 13px; color: #6B7280; margin-bottom: 5px; }}
    .ml-algo-label {{ font-size: 12px; font-weight: 600; color: #FF4D00; margin-bottom: 15px; display: block; }}
    .section-header {{ font-size: 16px; font-weight: 700; color: #333; margin-top: 20px; margin-bottom: 10px; border-bottom: 1px solid #E0E0E0; padding-bottom: 5px; }}
    h1, h2, h3 {{ color: #2C3E50; }}
    .dataframe-container {{ width: 100%; font-size: 13px; }}
</style>
""", unsafe_allow_html=True)

# ham load du lieu tu file csv
@st.cache_data
def load_data():
    try:
        logs = pd.read_csv('data/employee_logs.csv')
        survey = pd.read_csv('data/engagement_survey_raw.csv')
        staff = pd.read_csv('data/staff_info.csv')
        attrition = pd.read_csv('data/attrition_data.csv')
        return logs, survey, staff, attrition
    except FileNotFoundError:
        st.error("Data files not found. Ensure employee_logs.csv, engagement_survey_raw.csv, staff_info.csv, attrition_data.csv are in the directory.")
        st.stop()

# ham clean va tinh toan data
def process_data(logs, survey, staff, attrition):
    np.random.seed(42) # set seed de random ra giong nhau
    
    # xu ly log cham cong
    logs['Log_Date'] = pd.to_datetime(logs['Log_Date'])
    logs['Day_Name'] = logs['Log_Date'].dt.day_name()
    logs['Check_In_Dt'] = pd.to_datetime(logs['Log_Date'].astype(str) + ' ' + logs['Check_In_Time'], errors='coerce')
    logs['Check_Out_Dt'] = pd.to_datetime(logs['Log_Date'].astype(str) + ' ' + logs['Check_Out_Time'], errors='coerce')
    
    # tinh thoi gian checkin
    target_time = datetime.strptime("08:00:00", "%H:%M:%S").time()
    logs['Check_In_Time_Obj'] = logs['Check_In_Dt'].dt.time
    logs['Is_OnTime'] = logs['Check_In_Time_Obj'].apply(lambda x: 1 if x <= target_time else 0)
    logs['Is_Late'] = 1 - logs['Is_OnTime']
    
    # fix loi lam qua dem
    mask_night = logs['Check_Out_Dt'] < logs['Check_In_Dt']
    logs.loc[mask_night, 'Check_Out_Dt'] += pd.Timedelta(days=1)
    logs['Work_Hours'] = (logs['Check_Out_Dt'] - logs['Check_In_Dt']).dt.total_seconds() / 3600

    # tao data gia dinh cho model (vi data hien tai k du cot)
    driver_types = ['Chuyên lấy (Pick-up)', 'Chuyên giao (Delivery)', 'Kiêm nhiệm (Hybrid)']
    staff['Driver_Type'] = np.random.choice(driver_types, len(staff), p=[0.2, 0.3, 0.5])
    
    zone_types = ['Nội thành (Chung cư/Hẻm)', 'Ngoại thành (Đường lớn, Xa)']
    staff['Zone_Type'] = np.random.choice(zone_types, len(staff), p=[0.6, 0.4])
    
    # tinh income
    staff['Income_Per_Order'] = np.where(staff['Zone_Type'].str.contains('Nội thành'), 
                                         np.random.uniform(2000, 3500, len(staff)), 
                                         np.random.uniform(10000, 15000, len(staff)))

    # tao data cho kho
    wh_list = staff['Warehouse_ID'].dropna().unique()
    wh_mock = pd.DataFrame({
        'Warehouse_ID': wh_list,
        'Return_Rate': np.random.uniform(0.02, 0.15, len(wh_list)),
        'SLA_Delivery': np.random.uniform(0.85, 0.99, len(wh_list))
    })
    staff = staff.merge(wh_mock, on='Warehouse_ID', how='left')

    logs_merged = logs.merge(staff[['Staff_ID', 'Status', 'Role', 'Warehouse_ID', 'Driver_Type', 'Zone_Type']], on=['Staff_ID', 'Warehouse_ID'], how='left')

    # clean du lieu khao sat
    survey_clean = survey.copy()
    survey_clean['Employee_ID'] = survey_clean['Employee'].str.replace(r'Employee\s+', '', regex=True).str.strip()
    
    # map may cai label ve so
    mapping_q2 = {'Disagree': 1, 'Agree': 4, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    survey_clean['Q2_Score'] = survey_clean['Q2'].astype(str).map({str(k):v for k,v in mapping_q2.items()})
    mapping_happy = {'Low': 1, 'Medium': 3, 'High': 5}
    survey_clean['Happy_Score'] = survey_clean['Happiness'].map(mapping_happy)
    
    # fill na
    survey_clean['Q1_Score'] = pd.to_numeric(survey_clean['Q1_Score'], errors='coerce').fillna(survey_clean['Q1_Score'].median())
    survey_clean['Q2_Score'] = survey_clean['Q2_Score'].fillna(3)
    survey_clean['Happy_Score'] = survey_clean['Happy_Score'].fillna(3)
    survey_clean['Overall_Score'] = (survey_clean['Q1_Score'] + survey_clean['Q2_Score'] + survey_clean['Happy_Score']) / 3

    survey_clean = survey_clean.merge(staff[['Staff_ID', 'Department', 'Role']], left_on='Employee_ID', right_on='Staff_ID', how='left')

    # xu ly attrition
    attrition['Resign_Date'] = pd.to_datetime(attrition['Resign_Date'])
    attrition['Month_Year'] = attrition['Resign_Date'].dt.to_period('M').astype(str)
    
    # join bang
    attrition = attrition.merge(staff[['Staff_ID', 'Driver_Type', 'Zone_Type']], on='Staff_ID', how='left')

    return logs_merged, survey_clean, staff, attrition

# ham de set style cho chart theo chuan IBCS
def apply_ibcs_theme(fig, is_horizontal=False):
    layout_args = dict(
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Segoe UI", size=12, color=IBCS_ACTUAL),
        margin=dict(l=0, r=20, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_layout(**layout_args)
    if is_horizontal:
        fig.update_xaxes(showgrid=True, gridcolor="#F3F4F6", zeroline=False)
        fig.update_yaxes(showgrid=False, linecolor="#D1D5DB")
    else:
        fig.update_xaxes(showgrid=False, linecolor="#D1D5DB")
        fig.update_yaxes(showgrid=True, gridcolor="#F3F4F6", zeroline=False)
    return fig

# ham de train model random forest 
@st.cache_resource
def train_attrition_model(logs_df, survey_df, staff_df):
    
    # group by lay trung binh
    staff_logs = logs_df.groupby('Staff_ID').agg(
        Avg_Work_Hours=('Work_Hours', 'mean'),
        Total_Late=('Is_Late', 'sum')
    ).reset_index()

    # merge 3 cai df
    ml_df = staff_df[['Staff_ID', 'Status', 'Role']].merge(staff_logs, on='Staff_ID', how='left')
    ml_df = ml_df.merge(survey_df[['Staff_ID', 'Q1_Score', 'Q2_Score', 'Happy_Score']], on='Staff_ID', how='left')

    # dien may cho bi null
    ml_df = ml_df.fillna({
        'Avg_Work_Hours': 8.0, 
        'Total_Late': 0,
        'Q1_Score': 3.0, 
        'Q2_Score': 3.0, 
        'Happy_Score': 3.0
    })

    # set label
    ml_df['Is_Resigned'] = ml_df['Status'].apply(lambda x: 1 if x in ['Inactive', 'Resigned'] else 0)

    # chia X y ra de train
    features = ['Avg_Work_Hours', 'Total_Late', 'Q1_Score', 'Q2_Score', 'Happy_Score']
    X = ml_df[features]
    y = ml_df['Is_Resigned']

    # goi model
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X, y)
    
    # lay thong so importance
    if len(rf.classes_) > 1:
        importance_vals = rf.feature_importances_
    else:
        importance_vals = [0, 0, 0, 0, 0]

    feature_importances = pd.DataFrame({
        'Feature': ['Giờ làm trung bình', 'Tổng lần đi muộn', 'Điểm Công việc (Q1)', 'Điểm Môi trường (Q2)', 'Điểm Hạnh phúc'],
        'Importance': importance_vals
    }).sort_values('Importance', ascending=True)

    return rf, ml_df, features, feature_importances

# lay data
logs_raw, survey_raw, staff_raw, attrition_raw = load_data()
logs_df, survey_df, staff_df, attrition_df = process_data(logs_raw, survey_raw, staff_raw, attrition_raw)

# train model
rf_model, ml_dataset, ml_features, feature_importances = train_attrition_model(logs_df, survey_df, staff_df)


# sidebar 
st.sidebar.markdown("""
    <div style='text-align: center; padding: 15px 0; border-bottom: 2px solid #F0F0F0; margin-bottom: 20px;'>
        <h1 style='color: #FF4D00; margin: 0; font-size: 32px; font-weight: 900; letter-spacing: 1px;'>GHN</h1>
        <p style='color: #555; margin: 0; font-size: 13px; font-weight: 600; text-transform: uppercase;'>Operations Analytics</p>
    </div>
""", unsafe_allow_html=True)

# nut doi ngon ngu
lang_col1, lang_col2 = st.sidebar.columns(2)
if lang_col1.button("VN", use_container_width=True): st.session_state.lang = 'VN'
if lang_col2.button("ENG", use_container_width=True): st.session_state.lang = 'EN'

st.sidebar.markdown(f"<div style='font-size: 14px; font-weight: bold; margin-top: 20px; margin-bottom: 10px;'>{_('filter_title')}</div>", unsafe_allow_html=True)

# loc theo kho
all_warehouses = sorted(staff_df['Warehouse_ID'].dropna().unique())
all_wh_checkbox = st.sidebar.checkbox(_('all_wh'), value=True)
if all_wh_checkbox:
    selected_wh = all_warehouses
else:
    selected_wh = st.sidebar.multiselect(_('filter_wh'), all_warehouses, default=all_warehouses[:2] if len(all_warehouses) >= 2 else all_warehouses)

# loc theo role
all_roles = sorted(staff_df['Role'].dropna().unique())
all_role_checkbox = st.sidebar.checkbox(_('all_role'), value=True)
if all_role_checkbox:
    selected_roles = all_roles
else:
    selected_roles = st.sidebar.multiselect(_('filter_role'), all_roles, default=all_roles[:1] if len(all_roles) >= 1 else all_roles)

# loc theo ngay
st.sidebar.markdown(f"<div style='font-size: 13px; margin-top: 10px; margin-bottom: 5px;'>{_('filter_date')}</div>", unsafe_allow_html=True)
min_date = logs_df['Log_Date'].min().date()
max_date = logs_df['Log_Date'].max().date()

selected_date = st.sidebar.date_input(_('filter_date'), [min_date, max_date], label_visibility="collapsed")

# ap dung cai filter tren sidebar vao may cai df
logs_filtered = logs_df[(logs_df['Warehouse_ID'].isin(selected_wh)) & (logs_df['Role'].isin(selected_roles))]
attrition_filtered = attrition_df[(attrition_df['Warehouse_ID'].isin(selected_wh)) & (attrition_df['Role'].isin(selected_roles))]
staff_filtered = staff_df[(staff_df['Warehouse_ID'].isin(selected_wh)) & (staff_df['Role'].isin(selected_roles))]

if isinstance(selected_date, list) and len(selected_date) == 2:
    start_d, end_d = selected_date
    logs_filtered = logs_filtered[(logs_filtered['Log_Date'].dt.date >= start_d) & (logs_filtered['Log_Date'].dt.date <= end_d)]

# tinh cac kpi tong de show len
active_logs = logs_filtered[logs_filtered['Status'] == 'Active']
ontime_rate = (active_logs['Is_OnTime'].sum() / len(active_logs)) * 100 if len(active_logs) > 0 else 0
ontime_delta = ontime_rate - 98.0
overload_count = len(active_logs[active_logs['Work_Hours'] > 10])
overload_rate = (overload_count / len(active_logs)) * 100 if len(active_logs) > 0 else 0
overload_delta = overload_rate - 5.0

avg_hours = active_logs['Work_Hours'].mean() if len(active_logs) > 0 else 0
late_count = active_logs['Is_Late'].sum() if len(active_logs) > 0 else 0

total_resign = len(attrition_filtered)
total_active_staff = len(staff_filtered[staff_filtered['Status'] == 'Active'])
turnover_rate = (total_resign / (total_active_staff + total_resign)) * 100 if (total_active_staff + total_resign) > 0 else 0
avg_happy = survey_df['Overall_Score'].mean()


# header trang
st.title(_('app_title'))
st.markdown(_('app_sub'))
st.markdown("---")

# cac tab chinh
tab1, tab2, tab3, tab4, tab5 = st.tabs([_('tab_exec'), _('tab_ops'), _('tab_hr'), _('tab_data'), _('tab_ml')])

# Tab 1
with tab1:
    # 4 the hien thi sla
    st.markdown(f'<div class="section-header">{_("sla_title")}</div>', unsafe_allow_html=True)
    sla_col1, sla_col2, sla_col3, sla_col4 = st.columns(4)
    
    sla_col1.metric(_('sla_del'), "96.5%", "+1.5% vs Target", delta_color="normal")
    sla_col2.metric(_('sla_wh'), "98.2%", "+0.2% vs Target", delta_color="normal")
    sla_col3.metric(_('sla_att'), f"{ontime_rate:.1f}%", f"{ontime_delta:.1f}% vs Target (98%)", delta_color="normal")
    sla_col4.metric(_('sla_overload'), f"{overload_rate:.1f}%", f"{overload_delta:.1f}% vs Target (5%)", delta_color="inverse")

    # 4 the hien thi kpi hr
    st.markdown(f'<div class="section-header">{_("hr_title")}</div>', unsafe_allow_html=True)
    hr_col1, hr_col2, hr_col3, hr_col4 = st.columns(4)
    
    hr_col1.metric(_('kpi_staff'), f"{total_active_staff}", _('kpi_unit_p'))
    hr_col2.metric(_('turnover_rate'), f"{turnover_rate:.1f}%", _('kpi_alert'), delta_color="inverse")
    hr_col3.metric(_('kpi_resign'), f"{total_resign}", _('kpi_alert'), delta_color="inverse")
    hr_col4.metric(_('kpi_happy'), f"{avg_happy:.1f}/5.0", _('kpi_avg'))
    
    st.markdown("<br>", unsafe_allow_html=True)

    # hang 1
    r1_col1, r1_col2 = st.columns(2)
    
    with r1_col1:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">{_("c1_title")}</div><div class="ibcs-subtitle">{_("c1_sub")}</div>', unsafe_allow_html=True)
        
        ontime_count_val = int(active_logs['Is_OnTime'].sum())
        late_count_val = int(len(active_logs) - ontime_count_val)
        
        labels = [_('status_ontime'), _('status_late')]
        values = [ontime_count_val, late_count_val]
        
        fig1 = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values, 
            hole=0.5, 
            marker_colors=[IBCS_ACTUAL, IBCS_BAD],
            textinfo='label+percent',
            textposition='outside'
        )])
        fig1.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Segoe UI", size=12, color=IBCS_ACTUAL),
            margin=dict(l=20, r=20, t=10, b=20),
            showlegend=False
        )
        st.plotly_chart(fig1, use_container_width=True, height=250)
        st.markdown('</div>', unsafe_allow_html=True)

    with r1_col2:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">{_("c2_title")}</div><div class="ibcs-subtitle">{_("c2_sub")}</div>', unsafe_allow_html=True)
        wh_ontime = active_logs.groupby('Warehouse_ID')['Is_OnTime'].mean().reset_index()
        wh_ontime['Rate'] = wh_ontime['Is_OnTime'] * 100
        wh_ontime['Color'] = wh_ontime['Rate'].apply(lambda x: IBCS_BAD if x < 98 else IBCS_ACTUAL)
        wh_ontime = wh_ontime.sort_values('Rate')
        fig2 = go.Figure(go.Bar(y=wh_ontime['Warehouse_ID'], x=wh_ontime['Rate'], orientation='h', marker_color=wh_ontime['Color'], text=wh_ontime['Rate'].apply(lambda x: f"{x:.1f}%"), textposition='outside'))
        fig2.add_vline(x=98, line_dash="dash", line_color=IBCS_TARGET)
        fig2.update_xaxes(title=_('y_rate'))
        st.plotly_chart(apply_ibcs_theme(fig2, is_horizontal=True), use_container_width=True, height=250)
        st.markdown('</div>', unsafe_allow_html=True)

    # hang 2
    r2_col1, r2_col2 = st.columns(2)
    
    with r2_col1:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">{_("c3_title")}</div><div class="ibcs-subtitle">{_("c3_sub")}</div>', unsafe_allow_html=True)
        trend_resign = attrition_filtered.groupby('Month_Year').size().reset_index(name='Count')
        fig3 = go.Figure(go.Bar(x=trend_resign['Month_Year'], y=trend_resign['Count'], marker_color=IBCS_ACTUAL, text=trend_resign['Count'], textposition='outside'))
        st.plotly_chart(apply_ibcs_theme(fig3, is_horizontal=False), use_container_width=True, height=250)
        st.markdown('</div>', unsafe_allow_html=True)

    with r2_col2:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">{_("c4_title")}</div><div class="ibcs-subtitle">{_("c4_sub")}</div>', unsafe_allow_html=True)
        reasons = attrition_filtered['Resign_Reason'].value_counts().reset_index()
        reasons.columns = ['Reason', 'Count']
        fig4 = go.Figure(go.Bar(y=reasons['Reason'], x=reasons['Count'], orientation='h', marker_color=IBCS_ACTUAL, text=reasons['Count'], textposition='outside'))
        st.plotly_chart(apply_ibcs_theme(fig4, is_horizontal=True), use_container_width=True, height=250)
        st.markdown('</div>', unsafe_allow_html=True)

# Tab 2
with tab2:
    st.markdown(f'<div class="section-header">{_("ops_kpi_title")}</div>', unsafe_allow_html=True)
    op_col1, op_col2, op_col3, op_col4 = st.columns(4)
    
    op_col1.metric(_('kpi_ontime'), f"{ontime_rate:.1f}%", f"{ontime_delta:.1f}% vs Target", delta_color="normal")
    op_col2.metric(_('late_count'), f"{int(late_count)}", _('status_late'), delta_color="inverse")
    op_col3.metric(_('avg_hr'), f"{avg_hours:.1f}h", "Standard 8h", delta_color="inverse" if avg_hours>8 else "normal")
    op_col4.metric(_('sla_overload'), f"{overload_rate:.1f}%", f"{overload_delta:.1f}% vs Target", delta_color="inverse")
    st.markdown("<br>", unsafe_allow_html=True)

    # op row 1
    op_r1_col1, op_r1_col2 = st.columns(2)
    
    with op_r1_col1:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">{_("c5_title")}</div><div class="ibcs-subtitle">{_("c5_sub")}</div>', unsafe_allow_html=True)
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_stats = logs_filtered.groupby('Day_Name')['Is_Late'].mean().reset_index()
        day_stats['Rate'] = day_stats['Is_Late'] * 100
        day_stats['Day_Name'] = pd.Categorical(day_stats['Day_Name'], categories=day_order, ordered=True)
        fig_op1 = go.Figure(go.Bar(x=day_stats.sort_values('Day_Name')['Day_Name'], y=day_stats.sort_values('Day_Name')['Rate'], marker_color=IBCS_ACTUAL, text=day_stats.sort_values('Day_Name')['Rate'].apply(lambda x: f"{x:.1f}%"), textposition='outside'))
        fig_op1.update_yaxes(title=_('y_rate'))
        st.plotly_chart(apply_ibcs_theme(fig_op1, is_horizontal=False), use_container_width=True, height=300)
        st.markdown('</div>', unsafe_allow_html=True)

    with op_r1_col2:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">{_("c6_title")}</div><div class="ibcs-subtitle">{_("c6_sub")}</div>', unsafe_allow_html=True)
        staff_corr = active_logs.groupby('Staff_ID').agg({'Work_Hours': 'mean', 'Is_Late': 'sum'}).reset_index()
        fig_op2 = px.scatter(staff_corr, x='Work_Hours', y='Is_Late', trendline="ols")
        fig_op2.update_traces(marker=dict(size=8, color=IBCS_ACTUAL, opacity=0.5))
        if len(fig_op2.data) > 1: 
            fig_op2.data[1].line.color = IBCS_BAD
            
        fig_op2.update_layout(xaxis_title=_('x_workhr'), yaxis_title=_('y_late'))
        st.plotly_chart(apply_ibcs_theme(fig_op2, is_horizontal=False), use_container_width=True, height=300)
        st.markdown('</div>', unsafe_allow_html=True)

    # op row 2
    op_r2_col1, op_r2_col2 = st.columns(2)

    with op_r2_col1:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">{_("c13_title")}</div><div class="ibcs-subtitle">{_("c13_sub")}</div>', unsafe_allow_html=True)
        
        driver_hrs = active_logs.groupby('Driver_Type')['Work_Hours'].mean().reset_index().sort_values('Work_Hours')
        
        fig_op3 = go.Figure(go.Bar(
            y=driver_hrs['Driver_Type'], 
            x=driver_hrs['Work_Hours'], 
            orientation='h', 
            marker_color=[IBCS_BAD if x > 10 else IBCS_ACTUAL for x in driver_hrs['Work_Hours']], 
            text=driver_hrs['Work_Hours'].apply(lambda x: f"{x:.1f}h"), 
            textposition='outside'
        ))
        fig_op3.add_vline(x=8, line_dash="dash", line_color=IBCS_TARGET, annotation_text="Standard 8h")
        st.plotly_chart(apply_ibcs_theme(fig_op3, is_horizontal=True), use_container_width=True, height=300)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with op_r2_col2:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">{_("c14_title")}</div><div class="ibcs-subtitle">{_("c14_sub")}</div>', unsafe_allow_html=True)
        
        # summary data
        wh_summary = staff_filtered.groupby('Warehouse_ID').agg(
            Total_Staff=('Staff_ID', 'count'),
            Return_Rate=('Return_Rate', 'mean'),
            SLA_Delivery=('SLA_Delivery', 'mean')
        ).reset_index()
        
        # summary them
        attr_wh = attrition_filtered.groupby('Warehouse_ID').size().reset_index(name='Resign_Count')
        wh_summary = wh_summary.merge(attr_wh, on='Warehouse_ID', how='left').fillna(0)
        wh_summary['Turnover_Rate'] = (wh_summary['Resign_Count'] / wh_summary['Total_Staff']) * 100
        wh_summary['Return_Rate'] = wh_summary['Return_Rate'] * 100
        wh_summary['SLA_Delivery'] = wh_summary['SLA_Delivery'] * 100
        
        # the view
        wh_display = wh_summary[['Warehouse_ID', 'SLA_Delivery', 'Return_Rate', 'Turnover_Rate']].copy()
        wh_display.columns = ['Kho (Bưu Cục)', 'SLA Giao hàng (%)', 'Hoàn hàng (%)', 'Nghỉ việc (%)']
        
        st.dataframe(
            wh_display.style.background_gradient(subset=['Nghỉ việc (%)'], cmap="Reds")
                           .background_gradient(subset=['Hoàn hàng (%)'], cmap="Oranges")
                           .format({'SLA Giao hàng (%)': '{:.1f}%', 'Hoàn hàng (%)': '{:.1f}%', 'Nghỉ việc (%)': '{:.1f}%'}),
            use_container_width=True, 
            height=300
        )
        st.markdown('</div>', unsafe_allow_html=True)

# Tab 3
with tab3:
    st.markdown(f'<div class="section-header">{_("hr_kpi_title")}</div>', unsafe_allow_html=True)
    hr_m1, hr_m2, hr_m3, hr_m4 = st.columns(4)
    
    hr_m1.metric(_('kpi_staff'), f"{total_active_staff}", _('kpi_unit_p'))
    hr_m2.metric(_('turnover_rate'), f"{turnover_rate:.1f}%", _('kpi_alert'), delta_color="inverse")
    hr_m3.metric(_('kpi_resign'), f"{total_resign}", _('kpi_alert'), delta_color="inverse")
    hr_m4.metric(_('kpi_happy'), f"{avg_happy:.1f}/5.0", _('kpi_avg'))
    st.markdown("<br>", unsafe_allow_html=True)

    # hr row 1
    hr_r1_col1, hr_r1_col2 = st.columns(2)
    
    with hr_r1_col1:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">{_("c9_title")}</div><div class="ibcs-subtitle">{_("c9_sub")}</div>', unsafe_allow_html=True)
        if 'Department' in survey_df.columns:
            dept_happy = survey_df.groupby('Department')['Overall_Score'].mean().reset_index().sort_values('Overall_Score')
            fig_hr1 = go.Figure(go.Bar(y=dept_happy['Department'], x=dept_happy['Overall_Score'], orientation='h', marker_color=IBCS_ACTUAL, text=dept_happy['Overall_Score'].apply(lambda x: f"{x:.1f}"), textposition='outside'))
            fig_hr1.update_xaxes(range=[0, 5])
            st.plotly_chart(apply_ibcs_theme(fig_hr1, is_horizontal=True), use_container_width=True, height=250)
        st.markdown('</div>', unsafe_allow_html=True)

    with hr_r1_col2:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">{_("c10_title")}</div><div class="ibcs-subtitle">{_("c10_sub")}</div>', unsafe_allow_html=True)
        survey_avg = pd.DataFrame({'Metric': ['Q1 (Work)', 'Q2 (Environment)', 'Happiness'], 'Score': [survey_df['Q1_Score'].mean(), survey_df['Q2_Score'].mean(), survey_df['Happy_Score'].mean()]}).sort_values('Score')
        fig_hr2 = go.Figure(go.Bar(y=survey_avg['Metric'], x=survey_avg['Score'], orientation='h', marker_color=IBCS_ACTUAL, text=survey_avg['Score'].apply(lambda x: f"{x:.1f}"), textposition='outside'))
        fig_hr2.update_xaxes(range=[0, 5])
        st.plotly_chart(apply_ibcs_theme(fig_hr2, is_horizontal=True), use_container_width=True, height=250)
        st.markdown('</div>', unsafe_allow_html=True)

    # hr row 2
    hr_r2_col1, hr_r2_col2 = st.columns(2)

    with hr_r2_col1:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">{_("c11_title")}</div><div class="ibcs-subtitle">{_("c11_sub")}</div>', unsafe_allow_html=True)
        fig_hr3 = px.histogram(attrition_filtered, x="Tenure_Month", nbins=12, color_discrete_sequence=[IBCS_ACTUAL])
        fig_hr3.update_layout(xaxis_title="Tenure (Months)", yaxis_title="Count")
        st.plotly_chart(apply_ibcs_theme(fig_hr3, is_horizontal=False), use_container_width=True, height=250)
        st.markdown('</div>', unsafe_allow_html=True)

    with hr_r2_col2:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">{_("c15_title")}</div><div class="ibcs-subtitle">{_("c15_sub")}</div>', unsafe_allow_html=True)
        # gom df tinh rate
        zone_staff = staff_filtered.groupby('Zone_Type').size().reset_index(name='Total')
        zone_resign = attrition_filtered.groupby('Zone_Type').size().reset_index(name='Resigned')
        zone_turnover = zone_staff.merge(zone_resign, on='Zone_Type', how='left').fillna(0)
        zone_turnover['Turnover_Rate'] = (zone_turnover['Resigned'] / zone_turnover['Total']) * 100
        zone_turnover = zone_turnover.sort_values('Turnover_Rate', ascending=False)
        
        fig_hr5 = go.Figure(go.Bar(
            x=zone_turnover['Zone_Type'], 
            y=zone_turnover['Turnover_Rate'], 
            marker_color=[IBCS_BAD if x == zone_turnover['Turnover_Rate'].max() else IBCS_ACTUAL for x in zone_turnover['Turnover_Rate']], 
            text=zone_turnover['Turnover_Rate'].apply(lambda x: f"{x:.1f}%"), 
            textposition='outside'
        ))
        fig_hr5.update_yaxes(title="Turnover Rate (%)")
        st.plotly_chart(apply_ibcs_theme(fig_hr5, is_horizontal=False), use_container_width=True, height=250)
        st.markdown('</div>', unsafe_allow_html=True)

# Tab 4
with tab4:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Raw Data</div>', unsafe_allow_html=True)
        st.dataframe(survey_raw.head(8), use_container_width=True)
    with col2:
        st.markdown('<div class="section-header">Standardized Data</div>', unsafe_allow_html=True)
        st.dataframe(survey_df.head(8), use_container_width=True)

# Tab 5
with tab5:
    st.markdown(f'<div class="section-header">{_("ml_title")}</div>', unsafe_allow_html=True)
    
    ml_col1, ml_col2 = st.columns([1, 1.5])
    
    with ml_col1:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">{_("c16_title")}</div><div class="ibcs-subtitle">{_("c16_sub")}</div><div class="ml-algo-label">{_("ml_algo")}</div>', unsafe_allow_html=True)
        
        # ve chart ngang
        fig_ml1 = go.Figure(go.Bar(
            x=feature_importances['Importance'], 
            y=feature_importances['Feature'], 
            orientation='h', 
            marker_color=IBCS_ACTUAL
        ))
        fig_ml1.update_xaxes(title="Mức độ quan trọng (Feature Importance)")
        st.plotly_chart(apply_ibcs_theme(fig_ml1, is_horizontal=True), use_container_width=True, height=400)
        
        st.info(_("ml_insight"))
        
        # show mo ta cach model hoat dong
        with st.expander(_("ml_algo_desc_title")):
            st.markdown(_("ml_algo_desc"))
            
        st.markdown('</div>', unsafe_allow_html=True)

    with ml_col2:
        st.markdown(f'<div class="chart-container"><div class="ibcs-title">{_("c17_title")}</div><div class="ibcs-subtitle">{_("c17_sub")}</div><div class="ml-algo-label">{_("ml_algo")}</div>', unsafe_allow_html=True)
        
        # predict
        active_staff = ml_dataset[ml_dataset['Status'] == 'Active'].copy()
        
        if not active_staff.empty:
            probas = rf_model.predict_proba(active_staff[ml_features])
            if probas.shape[1] == 2:
                active_staff['Flight_Risk_Prob'] = probas[:, 1] * 100
            else:
                # truong hop data train toan active k co ai resign
                if rf_model.classes_[0] == 1:
                    active_staff['Flight_Risk_Prob'] = 100.0
                else:
                    active_staff['Flight_Risk_Prob'] = 0.0
            
            # lay top 10 nguy hiem nhat
            high_risk = active_staff.sort_values('Flight_Risk_Prob', ascending=False).head(10)
            
            high_risk_display = high_risk[['Staff_ID', 'Role', 'Flight_Risk_Prob', 'Avg_Work_Hours', 'Total_Late', 'Happy_Score']].copy()
            high_risk_display.columns = ['Mã NV', 'Vị trí', 'Nguy cơ Nghỉ việc (%)', 'Giờ làm (h)', 'Lần đi muộn', 'Điểm Hạnh phúc']
            
            st.dataframe(
                high_risk_display.style.background_gradient(subset=['Nguy cơ Nghỉ việc (%)'], cmap="Reds")
                               .format({'Nguy cơ Nghỉ việc (%)': '{:.1f}%', 'Giờ làm (h)': '{:.1f}', 'Điểm Hạnh phúc': '{:.1f}'}),
                use_container_width=True, 
                height=350
            )
            st.warning(_("ml_action"))
        else:
            st.write(_("ml_no_data"))
            
        st.markdown('</div>', unsafe_allow_html=True)