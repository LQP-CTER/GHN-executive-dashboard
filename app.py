import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="HR & Ops Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# IBCS COLOR SYSTEM
# Strict IBCS: black = actual, grey = reference, accent used SPARINGLY
# ══════════════════════════════════════════════════════════════════════════════
C_BLACK      = "#1A1A1A"   # Current / actual values
C_DARK_GREY  = "#4A4A4A"   # Secondary actual
C_MID_GREY   = "#8A8A8A"   # Reference / previous
C_LIGHT_GREY = "#D8D8D8"   # Grid lines, borders
C_BG         = "#F7F7F7"   # Page background
C_WHITE      = "#FFFFFF"   # Card background
C_ACCENT     = "#D0410A"   # Primary accent color — use max 1× per chart
C_WARN       = "#B54A00"   # Negative variance / alert (same family, darker)
C_GOOD       = "#2D6A2D"   # Positive variance / on-target

# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY IBCS THEME
# ══════════════════════════════════════════════════════════════════════════════
def apply_ibcs(fig, height=340, h_bars=False):
    """Apply strict IBCS chart styling."""
    fig.update_layout(
        template=None,
        paper_bgcolor=C_WHITE,
        plot_bgcolor=C_WHITE,
        font=dict(family="DM Sans, sans-serif", size=11, color=C_BLACK),
        title=None,
        height=height,
        margin=dict(l=8, r=16, t=8, b=8),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(size=10, family="DM Sans")
        ),
        hoverlabel=dict(bgcolor=C_WHITE, font_size=11, font_family="DM Sans"),
    )
    if h_bars:  # horizontal bar — gridlines on X, tick marks on Y
        fig.update_xaxes(
            showgrid=True, gridcolor=C_LIGHT_GREY, gridwidth=1,
            zeroline=False, linecolor=C_LIGHT_GREY,
            tickfont=dict(family="DM Mono, monospace", size=10, color=C_MID_GREY),
        )
        fig.update_yaxes(
            showgrid=False, zeroline=False, linecolor=C_LIGHT_GREY,
            ticks="outside", tickcolor=C_LIGHT_GREY, ticklen=3,
            tickfont=dict(family="DM Sans, sans-serif", size=10, color=C_BLACK),
        )
    else:       # vertical / scatter — gridlines on Y
        fig.update_xaxes(
            showgrid=False, zeroline=False, linecolor=C_LIGHT_GREY,
            ticks="outside", tickcolor=C_LIGHT_GREY, ticklen=3,
            tickfont=dict(family="DM Mono, monospace", size=10, color=C_MID_GREY),
        )
        fig.update_yaxes(
            showgrid=True, gridcolor=C_LIGHT_GREY, gridwidth=1,
            zeroline=False, linecolor=C_LIGHT_GREY,
            tickfont=dict(family="DM Mono, monospace", size=10, color=C_MID_GREY),
        )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# TRANSLATIONS
# ══════════════════════════════════════════════════════════════════════════════
TRANSLATIONS = {
    'VN': {
        'app_sub': "Báo cáo Hiệu suất Vận hành · Tuân thủ Chuẩn IBCS",
        'tab_exec': "A · Tổng quan", 'tab_ops': "B · Vận hành",
        'tab_hr': "C · Nhân sự", 'tab_data': "D · Dữ liệu", 'tab_ml': "E · Machine Learning",
        'filter_title': "BỘ LỌC",
        'all_wh': "Tất cả Kho", 'all_role': "Tất cả Vị trí",
        'filter_date': "Khoảng thời gian",
        'sla_title': "A · CHỈ SỐ SLA & VẬN HÀNH",
        'hr_title': "A2 · CHỈ SỐ NHÂN SỰ",
        'ops_kpi_title': "B · HIỆU SUẤT VẬN HÀNH",
        'hr_kpi_title': "C · NHÂN SỰ & TURNOVER",
        'sla_del': "SLA Giao hàng toàn trình", 'sla_wh': "SLA Xử lý tại Kho",
        'sla_att': "SLA Tuân thủ Giờ làm", 'sla_overload': "Ca quá tải (>10h)",
        'kpi_staff': "Tổng nhân sự Active", 'kpi_ontime': "Tỷ lệ Đúng giờ",
        'kpi_resign': "Nghỉ việc", 'kpi_happy': "Chỉ số Hạnh phúc",
        'kpi_unit_p': "nhân sự", 'kpi_alert': "Cảnh báo đỏ", 'kpi_avg': "Trung bình",
        'late_count': "Tổng lượt Đi muộn", 'avg_hr': "Giờ làm TB/Ngày",
        'turnover_rate': "Tỷ lệ Turnover",
        'c1_title': "Cơ cấu Tuân thủ Giờ làm",
        'c1_sub': "Tỷ trọng Đúng giờ vs Đi muộn · toàn bộ lượt Check-in",
        'c2_title': "WH01, WH05 vi phạm mục tiêu Đúng giờ",
        'c2_sub': "Tỷ lệ On-time (%) theo từng Kho · đường target = 98%",
        'c3_title': "Số lượng Nghỉ việc duy trì ở mức cao",
        'c3_sub': "Tổng nhân sự nghỉ việc theo tháng",
        'c4_title': "Vấn đề Quản lý là nguyên nhân gốc rễ",
        'c4_sub': "Phân bổ lý do nghỉ việc",
        'c5_title': "Thứ 2 và Chủ Nhật có tỷ lệ Đi muộn cao nhất",
        'c5_sub': "Phân tích Monday Blues · tỷ lệ đi muộn (%) theo thứ trong tuần",
        'c6_title': "Dấu hiệu kiệt sức: Làm thêm giờ → Đi muộn",
        'c6_sub': "Tương quan Giờ làm TB và Số lần Đi muộn / nhân sự",
        'c13_title': "Shipper Kiêm nhiệm chịu áp lực vận hành cao nhất",
        'c13_sub': "Giờ làm TB theo Loại hình Driver (Mô phỏng)",
        'c14_title': "Ma trận Sức khỏe Kho / Bưu cục",
        'c14_sub': "SLA · Hoàn hàng · Nghỉ việc theo từng Kho (Mô phỏng)",
        'c9_title': "Ops có Chỉ số Hạnh phúc thấp nhất",
        'c9_sub': "Hạnh phúc TB (thang 5) theo Phòng ban",
        'c10_title': "Môi trường làm việc đánh giá thấp hơn Nội dung công việc",
        'c10_sub': "So sánh điểm TB các thành phần khảo sát",
        'c11_title': "Nguy cơ chảy máu nhân sự: Nhóm 12–24 tháng",
        'c11_sub': "Phân bổ thâm niên (tháng) của nhóm đã nghỉ việc",
        'c15_title': "Tuyến Nội thành có Tỷ lệ Nghỉ việc cao nhất",
        'c15_sub': "Turnover Rate theo Đặc thù Tuyến giao hàng (Mô phỏng)",
        'ml_title': "E · DỰ ĐOÁN NGHỈ VIỆC · MACHINE LEARNING",
        'c16_title': "Random Forest: Yếu tố thúc đẩy Nghỉ việc",
        'c16_sub': "Feature Importance — mức độ ảnh hưởng đến quyết định nghỉ việc",
        'c17_title': "Flight Risk Radar: Nhân sự có Rủi ro cao",
        'c17_sub': "Xác suất nghỉ việc (%) của nhân sự Active · dựa trên dữ liệu hiện tại",
        'ml_algo': "Thuật toán: Random Forest Classifier",
        'ml_algo_desc_title': "Thuật toán hoạt động như thế nào?",
        'ml_algo_desc': "Random Forest hoạt động như một hội đồng chuyên gia: nhiều cây quyết định cùng phân tích dữ liệu và bỏ phiếu. Kết quả đa số là dự đoán cuối cùng.",
        'ml_insight': "**Insight:** Các yếu tố trên cùng là nguyên nhân chính dẫn đến quyết định nghỉ việc theo mô hình AI.",
        'ml_action': "**Đề xuất:** HR/Ops Manager cần 1-on-1 khẩn với nhân sự có nguy cơ > 60% để tìm hiểu tâm tư và giữ chân nhân tài.",
        'ml_no_data': "Không có dữ liệu nhân sự Active để dự đoán.",
        'y_rate': "Tỷ lệ (%)", 'x_workhr': "Giờ làm TB/Ngày", 'y_late': "Số lần Đi muộn",
        'status_ontime': "Đúng giờ", 'status_late': "Đi muộn", 'vs_target': "vs Target",
        'author_role': "Data Scientist & Analyst",
        'standard': "Tuân thủ IBCS",
    },
    'EN': {
        'app_sub': "Operational Performance Report · IBCS Compliant",
        'tab_exec': "A · Overview", 'tab_ops': "B · Operations",
        'tab_hr': "C · Attrition & HR", 'tab_data': "D · Data Quality", 'tab_ml': "E · Machine Learning",
        'filter_title': "FILTERS",
        'all_wh': "All Warehouses", 'all_role': "All Roles",
        'filter_date': "Period",
        'sla_title': "A · SLA & OPERATIONS METRICS",
        'hr_title': "A2 · WORKFORCE METRICS",
        'ops_kpi_title': "B · OPERATIONS KPIs",
        'hr_kpi_title': "C · HR & ATTRITION KPIs",
        'sla_del': "E2E Delivery SLA", 'sla_wh': "Warehouse Processing SLA",
        'sla_att': "Working Time Compliance SLA", 'sla_overload': "Overload Shift Rate (>10h)",
        'kpi_staff': "Total Active Staff", 'kpi_ontime': "On-time Arrival Rate",
        'kpi_resign': "Resignations", 'kpi_happy': "Happiness Score",
        'kpi_unit_p': "staff", 'kpi_alert': "Red Alert", 'kpi_avg': "Average",
        'late_count': "Total Late Check-ins", 'avg_hr': "Avg Work Hours/Day",
        'turnover_rate': "Turnover Rate",
        'c1_title': "Working Time Compliance Breakdown",
        'c1_sub': "On-time vs Late share · all check-in records",
        'c2_title': "WH01 and WH05 missed the on-time target",
        'c2_sub': "On-time rate (%) by Warehouse · target line = 98%",
        'c3_title': "Resignation numbers remain consistently high",
        'c3_sub': "Total resignations by month",
        'c4_title': "Management issues are the root cause",
        'c4_sub': "Resignation distribution by reason",
        'c5_title': "Mondays and Sundays see the highest late rates",
        'c5_sub': "Monday Blues Analysis · late rate (%) by weekday",
        'c6_title': "Burnout signal: Overtime hours → Late arrivals",
        'c6_sub': "Correlation between Avg Work Hours and Total Latenesses per Staff",
        'c13_title': "Hybrid Shippers carry the highest operational load",
        'c13_sub': "Avg Work Hours by Driver Type (Mock)",
        'c14_title': "Warehouse Health Matrix",
        'c14_sub': "SLA · Return Rate · Turnover by Warehouse (Mock)",
        'c9_title': "Operations dept has the lowest Happiness score",
        'c9_sub': "Avg Happiness Index (out of 5) by Department",
        'c10_title': "Environment rated lower than Work content",
        'c10_sub': "Average survey component scores company-wide",
        'c11_title': "Flight risk peaks at 12–24 months tenure",
        'c11_sub': "Tenure distribution (months) of resigned staff",
        'c15_title': "Urban (Apartment) routes have highest turnover",
        'c15_sub': "Turnover Rate by Delivery Zone (Mock)",
        'ml_title': "E · ATTRITION PREDICTIVE ANALYTICS · MACHINE LEARNING",
        'c16_title': "Random Forest: Turnover Drivers",
        'c16_sub': "Feature Importance scores — influence on resignation decisions",
        'c17_title': "Flight Risk Radar: High-risk Active Employees",
        'c17_sub': "Predicted resignation probability (%) based on current data",
        'ml_algo': "Algorithm: Random Forest Classifier",
        'ml_algo_desc_title': "How does this algorithm work?",
        'ml_algo_desc': "Random Forest operates like a council of experts: multiple decision trees analyze random data and vote, with the majority result as the final prediction.",
        'ml_insight': "**Insight:** Features at the top are the primary drivers of resignation decisions according to the AI model.",
        'ml_action': "**Actionable:** HR/Ops Managers should conduct urgent 1-on-1 meetings with staff showing > 60% flight risk to understand concerns and retain talent.",
        'ml_no_data': "No Active staff data available for prediction.",
        'y_rate': "Rate (%)", 'x_workhr': "Avg Work Hours/Day", 'y_late': "Times Late",
        'status_ontime': "On-time", 'status_late': "Late", 'vs_target': "vs Target",
        'author_role': "Data Scientist & Analyst",
        'standard': "IBCS Compliant",
    }
}

if 'lang' not in st.session_state:
    st.session_state.lang = 'VN'

def _(key):
    return TRANSLATIONS[st.session_state.lang].get(key, key)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS  —  mirroring Avocado dashboard aesthetic
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
/* ── RESET & BASE ── */
*, *::before, *::after {{ box-sizing: border-box; }}
html, body, .stApp {{
    background-color: {C_BG} !important;
    font-family: 'DM Sans', sans-serif !important;
    color: {C_BLACK};
}}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {{
    background-color: {C_WHITE} !important;
    border-right: 1px solid {C_LIGHT_GREY} !important;
}}
section[data-testid="stSidebar"] .block-container {{ padding: 0 1.25rem 1.5rem 1.25rem !important; }}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span:not(.material-symbols-rounded):not(.material-icons) {{
    color: {C_BLACK} !important;
    font-family: 'DM Sans', sans-serif !important;
}}

/* Sidebar top brand strip */
.sidebar-brand {{
    background: {C_BLACK};
    color: {C_WHITE};
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    padding: 14px 20px;
    margin: 0 -1.25rem 1.5rem -1.25rem;
    display: flex;
    align-items: center;
    gap: 10px;
}}
.sidebar-brand .brand-dot {{ color: {C_ACCENT}; font-size: 20px; line-height: 1; }}
.sidebar-brand .brand-sub {{ color: {C_MID_GREY}; font-weight: 400; font-size: 10px; letter-spacing: 0.1em; }}

/* Sidebar section label */
.sidebar-label {{
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: {C_MID_GREY};
    margin: 1.4rem 0 0.5rem 0;
    border-bottom: 1px solid {C_LIGHT_GREY};
    padding-bottom: 0.4rem;
    display: block;
}}

/* Language radio → pill toggle */
section[data-testid="stSidebar"] .stRadio > div {{
    display: flex !important;
    flex-direction: row !important;
    gap: 0 !important;
    background: {C_BG} !important;
    border: 1px solid {C_LIGHT_GREY} !important;
    border-radius: 4px !important;
    padding: 3px !important;
    margin-bottom: 8px !important;
}}
section[data-testid="stSidebar"] .stRadio label {{
    flex: 1 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    background: transparent !important;
    border-radius: 3px !important;
    padding: 6px 10px !important;
    cursor: pointer !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    color: {C_MID_GREY} !important;
    transition: all .15s ease !important;
    white-space: nowrap !important;
}}
section[data-testid="stSidebar"] .stRadio label:has(input:checked) {{
    background: {C_BLACK} !important;
    color: {C_WHITE} !important;
}}
section[data-testid="stSidebar"] .stRadio label:hover:not(:has(input:checked)) {{
    background: {C_LIGHT_GREY} !important;
    color: {C_BLACK} !important;
}}
section[data-testid="stSidebar"] .stRadio input[type="radio"] {{
    display: none !important;
}}

/* Sidebar info block */
.sidebar-info {{
    font-size: 11px;
    color: {C_DARK_GREY};
    line-height: 1.9;
    margin-top: 4px;
}}
.sidebar-info b {{ color: {C_BLACK}; font-weight: 600; }}
.sidebar-info small {{ color: {C_MID_GREY}; }}

/* ── REPORT HEADER ── */
.report-header {{
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    border-bottom: 3px solid {C_BLACK};
    padding-bottom: 14px;
    margin-bottom: 6px;
}}
.report-title {{
    font-size: 24px;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: {C_BLACK};
    line-height: 1.1;
}}
.report-title .accent {{ color: {C_ACCENT}; }}
.report-subtitle {{
    font-size: 13px;
    color: {C_MID_GREY};
    font-weight: 400;
    margin-top: 5px;
}}
.report-meta {{
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: {C_MID_GREY};
    text-align: right;
    line-height: 1.8;
}}

/* ── SECTION HEADER — black pill label (IBCS H1) ── */
.section-h {{
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: {C_WHITE};
    background: {C_BLACK};
    padding: 5px 12px;
    display: inline-block;
    margin: 28px 0 18px 0;
}}

/* ── KPI GRID — 1px separator style ── */
.kpi-grid {{
    display: grid;
    gap: 1px;
    background: {C_LIGHT_GREY};
    border: 1px solid {C_LIGHT_GREY};
    margin-bottom: 28px;
}}
.kpi-grid-4 {{ grid-template-columns: repeat(4, 1fr); }}
.kpi-grid-3 {{ grid-template-columns: repeat(3, 1fr); }}
.kpi-cell {{
    background: {C_WHITE};
    padding: 20px 22px;
}}
.kpi-label {{
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {C_MID_GREY};
    margin-bottom: 8px;
}}
.kpi-value {{
    font-family: 'DM Mono', monospace;
    font-size: 30px;
    font-weight: 500;
    color: {C_BLACK};
    letter-spacing: -0.04em;
    line-height: 1;
}}
.kpi-delta {{
    font-size: 11px;
    margin-top: 6px;
    font-family: 'DM Mono', monospace;
    color: {C_MID_GREY};
}}
.kpi-delta.pos {{ color: {C_GOOD}; }}
.kpi-delta.neg {{ color: {C_ACCENT}; }}

/* ── CHART CARD ── */
.card {{
    background: {C_WHITE};
    border: 1px solid {C_LIGHT_GREY};
    border-top: 3px solid {C_BLACK};
    padding: 16px 18px 10px 18px;
    margin-bottom: 18px;
    height: 100%;
}}
.card-title {{
    font-size: 13px;
    font-weight: 700;
    color: {C_BLACK};
    letter-spacing: -0.01em;
    margin-bottom: 3px;
    line-height: 1.3;
}}
.card-unit {{
    font-size: 11px;
    color: {C_MID_GREY};
    font-family: 'DM Mono', monospace;
    margin-bottom: 12px;
    padding-bottom: 10px;
    border-bottom: 1px solid {C_LIGHT_GREY};
}}

/* ── FOOTNOTE ── */
.footnote {{
    font-size: 10px;
    color: {C_MID_GREY};
    border-top: 1px solid {C_LIGHT_GREY};
    margin-top: 10px;
    padding-top: 6px;
    font-style: italic;
    font-family: 'DM Sans', sans-serif;
}}

/* ── IBCS TABLE ── */
.ibcs-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    margin-top: 4px;
}}
.ibcs-table th {{
    background: {C_BLACK};
    color: {C_WHITE};
    padding: 7px 10px;
    text-align: left;
    font-weight: 600;
    letter-spacing: 0.08em;
    font-size: 10px;
    text-transform: uppercase;
    font-family: 'DM Sans', sans-serif;
}}
.ibcs-table td {{
    padding: 6px 10px;
    border-bottom: 1px solid {C_LIGHT_GREY};
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: {C_DARK_GREY};
}}
.ibcs-table tr:hover td {{ background: #F2F2F2; }}

/* ── TABS ── */
div[data-testid="stTabs"] button {{
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: {C_MID_GREY} !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    padding: 10px 16px !important;
}}
div[data-testid="stTabs"] button[aria-selected="true"] {{
    color: {C_BLACK} !important;
    border-bottom-color: {C_BLACK} !important;
}}

/* ── HIDE DEFAULT METRICS ── */
div[data-testid="stMetric"] {{ display: none !important; }}

/* ── DATAFRAME ── */
.stDataFrame table {{
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
}}
.stDataFrame th {{
    background: {C_BLACK} !important;
    color: {C_WHITE} !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 10px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}}

/* ── ALERTS ── */
div[data-testid="stAlert"] {{
    border-radius: 0 !important;
    border-left-width: 4px !important;
    background: {C_WHITE} !important;
    font-size: 12px !important;
}}
div[data-testid="stExpander"] {{
    border: 1px solid {C_LIGHT_GREY} !important;
    border-radius: 0 !important;
    box-shadow: none !important;
}}

/* ── PREDICTION BOX ── */
.pred-box {{
    background: {C_BLACK};
    color: {C_WHITE};
    padding: 24px 28px;
    margin-top: 16px;
}}
.pred-label {{
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: {C_MID_GREY};
    margin-bottom: 8px;
}}
.pred-value {{
    font-family: 'DM Mono', monospace;
    font-size: 48px;
    font-weight: 500;
    color: {C_WHITE};
    letter-spacing: -0.04em;
}}
.pred-context {{
    font-size: 11px;
    color: {C_MID_GREY};
    margin-top: 6px;
}}

/* ── MISC ── */
hr {{ border: none; border-top: 1px solid {C_LIGHT_GREY}; margin: 16px 0; }}
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: {C_BG}; }}
::-webkit-scrollbar-thumb {{ background: {C_LIGHT_GREY}; border-radius: 0; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    try:
        logs      = pd.read_csv('data/employee_logs.csv')
        survey    = pd.read_csv('data/engagement_survey_raw.csv')
        staff     = pd.read_csv('data/staff_info.csv')
        attrition = pd.read_csv('data/attrition_data.csv')
        return logs, survey, staff, attrition
    except FileNotFoundError:
        st.error("Data files not found in /data directory.")
        st.stop()

def process_data(logs, survey, staff, attrition):
    np.random.seed(42)
    logs['Log_Date']     = pd.to_datetime(logs['Log_Date'])
    logs['Day_Name']     = logs['Log_Date'].dt.day_name()
    logs['Check_In_Dt']  = pd.to_datetime(logs['Log_Date'].astype(str) + ' ' + logs['Check_In_Time'],  errors='coerce')
    logs['Check_Out_Dt'] = pd.to_datetime(logs['Log_Date'].astype(str) + ' ' + logs['Check_Out_Time'], errors='coerce')
    target_time          = datetime.strptime("08:00:00", "%H:%M:%S").time()
    logs['Check_In_Time_Obj'] = logs['Check_In_Dt'].dt.time
    logs['Is_OnTime']    = logs['Check_In_Time_Obj'].apply(lambda x: 1 if x <= target_time else 0)
    logs['Is_Late']      = 1 - logs['Is_OnTime']
    mask_night           = logs['Check_Out_Dt'] < logs['Check_In_Dt']
    logs.loc[mask_night, 'Check_Out_Dt'] += pd.Timedelta(days=1)
    logs['Work_Hours']   = (logs['Check_Out_Dt'] - logs['Check_In_Dt']).dt.total_seconds() / 3600

    driver_types         = ['Chuyên lấy (Pick-up)', 'Chuyên giao (Delivery)', 'Kiêm nhiệm (Hybrid)']
    staff['Driver_Type'] = np.random.choice(driver_types, len(staff), p=[0.2, 0.3, 0.5])
    zone_types           = ['Nội thành (Chung cư/Hẻm)', 'Ngoại thành (Đường lớn, Xa)']
    staff['Zone_Type']   = np.random.choice(zone_types, len(staff), p=[0.6, 0.4])
    wh_list              = staff['Warehouse_ID'].dropna().unique()
    wh_mock              = pd.DataFrame({
        'Warehouse_ID': wh_list,
        'Return_Rate':  np.random.uniform(0.02, 0.15, len(wh_list)),
        'SLA_Delivery': np.random.uniform(0.85, 0.99, len(wh_list))
    })
    staff                = staff.merge(wh_mock, on='Warehouse_ID', how='left')
    logs_merged          = logs.merge(
        staff[['Staff_ID','Status','Role','Warehouse_ID','Driver_Type','Zone_Type']],
        on=['Staff_ID','Warehouse_ID'], how='left'
    )

    sc                   = survey.copy()
    sc['Employee_ID']    = sc['Employee'].str.replace(r'Employee\s+', '', regex=True).str.strip()
    mq2                  = {'Disagree':1,'Agree':4,'1':1,'2':2,'3':3,'4':4,'5':5,1:1,2:2,3:3,4:4,5:5}
    sc['Q2_Score']       = sc['Q2'].astype(str).map({str(k):v for k,v in mq2.items()})
    sc['Happy_Score']    = sc['Happiness'].map({'Low':1,'Medium':3,'High':5})
    sc['Q1_Score']       = pd.to_numeric(sc['Q1_Score'], errors='coerce').fillna(sc['Q1_Score'].median() if 'Q1_Score' in sc else 3)
    sc['Q2_Score']       = sc['Q2_Score'].fillna(3)
    sc['Happy_Score']    = sc['Happy_Score'].fillna(3)
    sc['Overall_Score']  = (sc['Q1_Score'] + sc['Q2_Score'] + sc['Happy_Score']) / 3
    sc                   = sc.merge(staff[['Staff_ID','Department','Role']], left_on='Employee_ID', right_on='Staff_ID', how='left')

    attrition['Resign_Date'] = pd.to_datetime(attrition['Resign_Date'])
    attrition['Month_Year']  = attrition['Resign_Date'].dt.to_period('M').astype(str)
    attrition                = attrition.merge(staff[['Staff_ID','Driver_Type','Zone_Type']], on='Staff_ID', how='left')
    return logs_merged, sc, staff, attrition

@st.cache_resource
def train_model(logs_df, survey_df, staff_df):
    sl     = logs_df.groupby('Staff_ID').agg(Avg_Work_Hours=('Work_Hours','mean'), Total_Late=('Is_Late','sum')).reset_index()
    ml     = staff_df[['Staff_ID','Status','Role']].merge(sl, on='Staff_ID', how='left')
    ml     = ml.merge(survey_df[['Staff_ID','Q1_Score','Q2_Score','Happy_Score']], on='Staff_ID', how='left')
    ml     = ml.fillna({'Avg_Work_Hours':8.0,'Total_Late':0,'Q1_Score':3.0,'Q2_Score':3.0,'Happy_Score':3.0})
    ml['Is_Resigned'] = ml['Status'].apply(lambda x: 1 if x in ['Inactive','Resigned'] else 0)
    feats  = ['Avg_Work_Hours','Total_Late','Q1_Score','Q2_Score','Happy_Score']
    rf     = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(ml[feats], ml['Is_Resigned'])
    imp    = rf.feature_importances_ if len(rf.classes_) > 1 else [0]*5
    fi_df  = pd.DataFrame({
        'Feature':    ['Giờ làm trung bình','Tổng lần đi muộn','Điểm Công việc (Q1)','Điểm Môi trường (Q2)','Điểm Hạnh phúc'],
        'Feature_EN': ['Avg Work Hours','Total Late Count','Work Score (Q1)','Environment Score (Q2)','Happiness Score'],
        'Importance': imp
    }).sort_values('Importance', ascending=True)
    return rf, ml, feats, fi_df

# ── Load data ──
logs_raw, survey_raw, staff_raw, attrition_raw = load_data()
logs_df, survey_df, staff_df, attrition_df     = process_data(logs_raw, survey_raw, staff_raw, attrition_raw)
rf_model, ml_dataset, ml_features, feat_imp    = train_model(logs_df, survey_df, staff_df)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span class="brand-dot">●</span>
        <div>
            OPS ANALYTICS
            <div class="brand-sub">Operations & HR Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Language pill toggle
    st.markdown('<span class="sidebar-label">Language</span>', unsafe_allow_html=True)
    lang_opts   = {"Tiếng Việt": "VN", "English": "EN"}
    lang_labels = list(lang_opts.keys())
    cur_label   = lang_labels[0] if st.session_state.lang == 'VN' else lang_labels[1]
    sel         = st.radio("lang", lang_labels, index=lang_labels.index(cur_label),
                           label_visibility="collapsed", horizontal=True, key="lang_r")
    if lang_opts[sel] != st.session_state.lang:
        st.session_state.lang = lang_opts[sel]; st.rerun()

    # Filters
    st.markdown(f'<span class="sidebar-label">{_("filter_title")}</span>', unsafe_allow_html=True)
    all_wh   = sorted(staff_df['Warehouse_ID'].dropna().unique())
    all_role = sorted(staff_df['Role'].dropna().unique())
    wh_all   = st.checkbox(_('all_wh'),   value=True)
    sel_wh   = all_wh   if wh_all   else st.multiselect("Kho", all_wh,   default=all_wh[:2])
    role_all = st.checkbox(_('all_role'), value=True)
    sel_role = all_role if role_all else st.multiselect("Vị trí", all_role, default=all_role[:1])

    st.markdown(f'<span class="sidebar-label">{_("filter_date")}</span>', unsafe_allow_html=True)
    min_d = logs_df['Log_Date'].min().date()
    max_d = logs_df['Log_Date'].max().date()
    sel_date = st.date_input("Period", [min_d, max_d], label_visibility="collapsed")

    st.markdown(f"""
    <div style="margin-top: 32px; padding-top: 16px; border-top: 1px solid {C_LIGHT_GREY};">
        <div class="sidebar-info">
            <b>Author</b><br>Lê Quý Phát<br>
            <small>{_('author_role')}</small><br><br>
            <b>Standard</b><br>{_('standard')}<br>
            <small>International Business<br>Communication Standards</small>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Apply filters ──
lf  = logs_df[(logs_df['Warehouse_ID'].isin(sel_wh)) & (logs_df['Role'].isin(sel_role))]
af  = attrition_df[(attrition_df['Warehouse_ID'].isin(sel_wh)) & (attrition_df['Role'].isin(sel_role))]
sf  = staff_df[(staff_df['Warehouse_ID'].isin(sel_wh)) & (staff_df['Role'].isin(sel_role))]
if isinstance(sel_date, (list, tuple)) and len(sel_date) == 2:
    lf = lf[(lf['Log_Date'].dt.date >= sel_date[0]) & (lf['Log_Date'].dt.date <= sel_date[1])]

# ── KPI aggregates ──
al            = lf[lf['Status'] == 'Active']
n_al          = len(al)
ontime_rate   = al['Is_OnTime'].sum() / n_al * 100 if n_al else 0
ontime_delta  = ontime_rate - 98.0
overload_rate = len(al[al['Work_Hours'] > 10]) / n_al * 100 if n_al else 0
overload_d    = overload_rate - 5.0
avg_hours     = al['Work_Hours'].mean() if n_al else 0
late_count    = int(al['Is_Late'].sum()) if n_al else 0
total_resign  = len(af)
total_active  = len(sf[sf['Status'] == 'Active'])
denom         = total_active + total_resign
turnover_rate = total_resign / denom * 100 if denom else 0
avg_happy     = survey_df['Overall_Score'].mean()


# ══════════════════════════════════════════════════════════════════════════════
# REPORT HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="report-header">
    <div>
        <div class="report-title">
            <span class="accent">Enterprise</span> Executive Dashboard — Management Report
        </div>
        <div class="report-subtitle">{_('app_sub')}</div>
    </div>
    <div class="report-meta">
        IBCS-compliant &nbsp;|&nbsp; {n_al:,} active log records<br>
        Warehouses: {len(sel_wh)} &nbsp;|&nbsp; Roles: {len(sel_role)}
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    _('tab_exec'), _('tab_ops'), _('tab_hr'), _('tab_data'), _('tab_ml')
])

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def card_open(label, unit):
    return f'<div class="card"><div class="card-title">{label}</div><div class="card-unit">{unit}</div>'

def card_foot(note):
    return f'<div class="footnote">{note}</div></div>'


# ══════════════════════════════════════════════════════════════════════════════
# TAB A — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    # ── SLA KPI grid ─────────────────────────────────────────────────────────
    st.markdown(f'<div class="section-h">{_("sla_title")}</div>', unsafe_allow_html=True)

    od_cls = "pos" if ontime_delta >= 0 else "neg"
    or_cls = "neg" if overload_d  > 0  else "pos"

    st.markdown(f"""
    <div class="kpi-grid kpi-grid-4">
        <div class="kpi-cell">
            <div class="kpi-label">{_('sla_del')}</div>
            <div class="kpi-value">96.5%</div>
            <div class="kpi-delta pos">▲ +1.5% {_('vs_target')}</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-label">{_('sla_wh')}</div>
            <div class="kpi-value">98.2%</div>
            <div class="kpi-delta pos">▲ +0.2% {_('vs_target')}</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-label">{_('sla_att')}</div>
            <div class="kpi-value">{ontime_rate:.1f}%</div>
            <div class="kpi-delta {od_cls}">{'▲' if ontime_delta>=0 else '▼'} {ontime_delta:+.1f}% {_('vs_target')} (98%)</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-label">{_('sla_overload')}</div>
            <div class="kpi-value">{overload_rate:.1f}%</div>
            <div class="kpi-delta {or_cls}">{'▼' if overload_d<=0 else '▲'} {overload_d:+.1f}% {_('vs_target')} (5%)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── HR KPI grid ───────────────────────────────────────────────────────────
    st.markdown(f'<div class="section-h">{_("hr_title")}</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="kpi-grid kpi-grid-4">
        <div class="kpi-cell">
            <div class="kpi-label">{_('kpi_staff')}</div>
            <div class="kpi-value">{total_active:,}</div>
            <div class="kpi-delta">{_('kpi_unit_p')}</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-label">{_('turnover_rate')}</div>
            <div class="kpi-value">{turnover_rate:.1f}%</div>
            <div class="kpi-delta neg">▲ {_('kpi_alert')}</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-label">{_('kpi_resign')}</div>
            <div class="kpi-value">{total_resign:,}</div>
            <div class="kpi-delta neg">▲ {_('kpi_alert')}</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-label">{_('kpi_happy')}</div>
            <div class="kpi-value">{avg_happy:.1f}<span style="font-size:16px;color:{C_MID_GREY};">/5.0</span></div>
            <div class="kpi-delta">{_('kpi_avg')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Charts row 1 ─────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(card_open(_('c1_title'), _('c1_sub')), unsafe_allow_html=True)
        ot_v = int(al['Is_OnTime'].sum()); la_v = int(n_al - ot_v)
        fig1 = go.Figure(go.Pie(
            labels=[_('status_ontime'), _('status_late')],
            values=[ot_v, la_v], hole=0.60,
            marker_colors=[C_BLACK, C_ACCENT],
            textinfo='label+percent', textposition='outside',
            textfont=dict(size=11, family='DM Sans'),
        ))
        fig1.add_annotation(
            text=f"<b style='font-family:DM Mono'>{ot_v+la_v:,}</b><br><span style='font-size:9px;color:{C_MID_GREY}'>TOTAL</span>",
            x=0.5, y=0.5, showarrow=False, font=dict(size=13, family='DM Mono', color=C_BLACK)
        )
        fig1.update_layout(template=None, paper_bgcolor=C_WHITE, plot_bgcolor=C_WHITE,
                           showlegend=False, margin=dict(l=60, r=60, t=16, b=16), height=300,
                           font=dict(family='DM Sans'))
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown(card_foot("IBCS: solid = actual · donut centre = total volume"), unsafe_allow_html=True)

    with c2:
        st.markdown(card_open(_('c2_title'), _('c2_sub')), unsafe_allow_html=True)
        wh_ot        = al.groupby('Warehouse_ID')['Is_OnTime'].mean().reset_index()
        wh_ot['Rate']= wh_ot['Is_OnTime'] * 100
        wh_ot['C']   = wh_ot['Rate'].apply(lambda x: C_ACCENT if x < 98 else C_BLACK)
        wh_ot        = wh_ot.sort_values('Rate')
        fig2 = go.Figure(go.Bar(
            y=wh_ot['Warehouse_ID'], x=wh_ot['Rate'], orientation='h',
            marker_color=wh_ot['C'],
            text=wh_ot['Rate'].apply(lambda x: f"{x:.1f}%"), textposition='outside',
            textfont=dict(size=10, family='DM Mono')
        ))
        fig2.add_vline(x=98, line_dash="dot", line_color=C_MID_GREY, line_width=1.5,
                       annotation_text="Target 98%",
                       annotation_font=dict(size=9, color=C_MID_GREY),
                       annotation_position="top right")
        fig2.update_xaxes(range=[80, 106], ticksuffix="%")
        apply_ibcs(fig2, height=300, h_bars=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown(card_foot("▪ Red bar = below 98% target · dotted line = target threshold"), unsafe_allow_html=True)

    # ── Charts row 2 ─────────────────────────────────────────────────────────
    c3, c4 = st.columns(2)

    with c3:
        st.markdown(card_open(_('c3_title'), _('c3_sub')), unsafe_allow_html=True)
        trend = af.groupby('Month_Year').size().reset_index(name='Count')
        fig3  = go.Figure(go.Bar(
            x=trend['Month_Year'], y=trend['Count'],
            marker_color=C_BLACK,
            text=trend['Count'], textposition='outside',
            textfont=dict(size=10, family='DM Mono')
        ))
        apply_ibcs(fig3, height=300)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown(card_foot("Monthly resignation count · all selected warehouses & roles"), unsafe_allow_html=True)

    with c4:
        st.markdown(card_open(_('c4_title'), _('c4_sub')), unsafe_allow_html=True)
        reasons = af['Resign_Reason'].value_counts().reset_index()
        reasons.columns = ['Reason', 'Count']
        reasons = reasons.sort_values('Count')
        top_r   = reasons.iloc[-1]['Reason']
        fig4 = go.Figure(go.Bar(
            y=reasons['Reason'], x=reasons['Count'], orientation='h',
            marker_color=[C_ACCENT if r == top_r else C_DARK_GREY for r in reasons['Reason']],
            text=reasons['Count'], textposition='outside',
            textfont=dict(size=10, family='DM Mono')
        ))
        apply_ibcs(fig4, height=300, h_bars=True)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown(card_foot("▪ Red = leading cause · ranked ascending by count"), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB B — OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f'<div class="section-h">{_("ops_kpi_title")}</div>', unsafe_allow_html=True)

    oh_cls = "neg" if avg_hours > 8 else "pos"
    st.markdown(f"""
    <div class="kpi-grid kpi-grid-4">
        <div class="kpi-cell">
            <div class="kpi-label">{_('kpi_ontime')}</div>
            <div class="kpi-value">{ontime_rate:.1f}%</div>
            <div class="kpi-delta {od_cls}">{'▲' if ontime_delta>=0 else '▼'} {ontime_delta:+.1f}% {_('vs_target')}</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-label">{_('late_count')}</div>
            <div class="kpi-value">{late_count:,}</div>
            <div class="kpi-delta neg">▲ {_('kpi_alert')}</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-label">{_('avg_hr')}</div>
            <div class="kpi-value">{avg_hours:.1f}h</div>
            <div class="kpi-delta {oh_cls}">Standard 8h</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-label">{_('sla_overload')}</div>
            <div class="kpi-value">{overload_rate:.1f}%</div>
            <div class="kpi-delta {or_cls}">{'▲' if overload_d>0 else '▼'} {overload_d:+.1f}% {_('vs_target')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    b1, b2 = st.columns(2)

    with b1:
        st.markdown(card_open(_('c5_title'), _('c5_sub')), unsafe_allow_html=True)
        day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        ds = lf.groupby('Day_Name')['Is_Late'].mean().reset_index()
        ds['Rate']     = ds['Is_Late'] * 100
        ds['Day_Name'] = pd.Categorical(ds['Day_Name'], categories=day_order, ordered=True)
        ds = ds.sort_values('Day_Name')
        peak = ds.loc[ds['Rate'].idxmax(), 'Day_Name']
        fig5 = go.Figure(go.Bar(
            x=ds['Day_Name'], y=ds['Rate'],
            marker_color=[C_ACCENT if d == peak else C_BLACK for d in ds['Day_Name']],
            text=ds['Rate'].apply(lambda x: f"{x:.1f}%"), textposition='outside',
            textfont=dict(size=10, family='DM Mono')
        ))
        fig5.update_yaxes(ticksuffix="%")
        apply_ibcs(fig5, height=320)
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown(card_foot("▪ Red bar = peak day · IBCS: black bars = actual values"), unsafe_allow_html=True)

    with b2:
        st.markdown(card_open(_('c6_title'), _('c6_sub')), unsafe_allow_html=True)
        sc_df = al.groupby('Staff_ID').agg({'Work_Hours':'mean','Is_Late':'sum'}).reset_index()
        
        fig6  = px.scatter(sc_df, x='Work_Hours', y='Is_Late',
                           color_discrete_sequence=[C_BLACK], opacity=0.25)
        
        # Update marker size for scatter
        if len(fig6.data) > 0:
            fig6.data[0].marker.size = 6

        # Manually calculate and add OLS trendline using numpy (avoids statsmodels dependency)
        valid_data = sc_df.dropna(subset=['Work_Hours', 'Is_Late'])
        if len(valid_data) > 1:
            z = np.polyfit(valid_data['Work_Hours'], valid_data['Is_Late'], 1)
            p = np.poly1d(z)
            x_trend = np.array([valid_data['Work_Hours'].min(), valid_data['Work_Hours'].max()])
            y_trend = p(x_trend)
            fig6.add_trace(go.Scatter(
                x=x_trend, y=y_trend, mode='lines', 
                line=dict(color=C_ACCENT, width=2), showlegend=False, hoverinfo='skip'
            ))

        fig6.update_xaxes(title_text=_('x_workhr'))
        fig6.update_yaxes(title_text=_('y_late'))
        apply_ibcs(fig6, height=320)
        st.plotly_chart(fig6, use_container_width=True)
        st.markdown(card_foot("OLS trendline · top 5% volume outliers included · each dot = 1 staff member"), unsafe_allow_html=True)

    b3, b4 = st.columns(2)

    with b3:
        st.markdown(card_open(_('c13_title'), _('c13_sub')), unsafe_allow_html=True)
        dh = al.groupby('Driver_Type')['Work_Hours'].mean().reset_index().sort_values('Work_Hours')
        fig13 = go.Figure(go.Bar(
            y=dh['Driver_Type'], x=dh['Work_Hours'], orientation='h',
            marker_color=[C_ACCENT if x > 9.5 else C_BLACK for x in dh['Work_Hours']],
            text=dh['Work_Hours'].apply(lambda x: f"{x:.1f}h"), textposition='outside',
            textfont=dict(size=10, family='DM Mono')
        ))
        fig13.add_vline(x=8, line_dash="dot", line_color=C_MID_GREY, line_width=1.5,
                        annotation_text="Standard 8h",
                        annotation_font=dict(size=9, color=C_MID_GREY))
        apply_ibcs(fig13, height=300, h_bars=True)
        st.plotly_chart(fig13, use_container_width=True)
        st.markdown(card_foot("Mock data · dotted = 8h standard · red = overload group"), unsafe_allow_html=True)

    with b4:
        st.markdown(card_open(_('c14_title'), _('c14_sub')), unsafe_allow_html=True)
        ws = sf.groupby('Warehouse_ID').agg(
            Total=('Staff_ID','count'), Return=('Return_Rate','mean'), SLA=('SLA_Delivery','mean')
        ).reset_index()
        aw = af.groupby('Warehouse_ID').size().reset_index(name='Resigned')
        ws = ws.merge(aw, on='Warehouse_ID', how='left').fillna(0)
        ws['Turnover'] = ws['Resigned'] / ws['Total'] * 100
        ws['Return']   = ws['Return'] * 100
        ws['SLA']      = ws['SLA'] * 100
        wd = ws[['Warehouse_ID','SLA','Return','Turnover']].copy()
        wd.columns = ['Kho','SLA Giao hàng (%)','Hoàn hàng (%)','Nghỉ việc (%)']
        
        # Using Streamlit's native column_config to format without matplotlib dependency
        st.dataframe(
            wd,
            column_config={
                "SLA Giao hàng (%)": st.column_config.NumberColumn(format="%.1f%%"),
                "Hoàn hàng (%)": st.column_config.NumberColumn(format="%.1f%%"),
                "Nghỉ việc (%)": st.column_config.NumberColumn(format="%.1f%%"),
            },
            use_container_width=True, 
            height=280, 
            hide_index=True
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB C — HR & ATTRITION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f'<div class="section-h">{_("hr_kpi_title")}</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="kpi-grid kpi-grid-4">
        <div class="kpi-cell">
            <div class="kpi-label">{_('kpi_staff')}</div>
            <div class="kpi-value">{total_active:,}</div>
            <div class="kpi-delta">{_('kpi_unit_p')}</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-label">{_('turnover_rate')}</div>
            <div class="kpi-value">{turnover_rate:.1f}%</div>
            <div class="kpi-delta neg">▲ {_('kpi_alert')}</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-label">{_('kpi_resign')}</div>
            <div class="kpi-value">{total_resign:,}</div>
            <div class="kpi-delta neg">▲ {_('kpi_alert')}</div>
        </div>
        <div class="kpi-cell">
            <div class="kpi-label">{_('kpi_happy')}</div>
            <div class="kpi-value">{avg_happy:.1f}<span style="font-size:16px;color:{C_MID_GREY};">/5.0</span></div>
            <div class="kpi-delta">{_('kpi_avg')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    h1, h2 = st.columns(2)

    with h1:
        st.markdown(card_open(_('c9_title'), _('c9_sub')), unsafe_allow_html=True)
        if 'Department' in survey_df.columns:
            dh = survey_df.groupby('Department')['Overall_Score'].mean().reset_index().sort_values('Overall_Score')
            mn = dh.iloc[0]['Department']
            fig9 = go.Figure(go.Bar(
                y=dh['Department'], x=dh['Overall_Score'], orientation='h',
                marker_color=[C_ACCENT if d == mn else C_BLACK for d in dh['Department']],
                text=dh['Overall_Score'].apply(lambda x: f"{x:.1f}"), textposition='outside',
                textfont=dict(size=10, family='DM Mono')
            ))
            fig9.update_xaxes(range=[0, 5.8])
            fig9.add_vline(x=3, line_dash="dot", line_color=C_MID_GREY, line_width=1,
                           annotation_text="Mid 3.0", annotation_font=dict(size=9, color=C_MID_GREY))
            apply_ibcs(fig9, height=300, h_bars=True)
            st.plotly_chart(fig9, use_container_width=True)
        st.markdown(card_foot("Scale: 1–5 · red = lowest scoring department"), unsafe_allow_html=True)

    with h2:
        st.markdown(card_open(_('c10_title'), _('c10_sub')), unsafe_allow_html=True)
        sa = pd.DataFrame({
            'Metric': ['Q1 (Work)','Q2 (Environment)','Happiness'],
            'Score':  [survey_df['Q1_Score'].mean(), survey_df['Q2_Score'].mean(), survey_df['Happy_Score'].mean()]
        }).sort_values('Score')
        mn2 = sa.iloc[0]['Metric']
        fig10 = go.Figure(go.Bar(
            y=sa['Metric'], x=sa['Score'], orientation='h',
            marker_color=[C_ACCENT if m == mn2 else C_BLACK for m in sa['Metric']],
            text=sa['Score'].apply(lambda x: f"{x:.2f}"), textposition='outside',
            textfont=dict(size=10, family='DM Mono')
        ))
        fig10.update_xaxes(range=[0, 5.8])
        apply_ibcs(fig10, height=300, h_bars=True)
        st.plotly_chart(fig10, use_container_width=True)
        st.markdown(card_foot("Company-wide average scores · scale 1–5 · red = lowest component"), unsafe_allow_html=True)

    h3, h4 = st.columns(2)

    with h3:
        st.markdown(card_open(_('c11_title'), _('c11_sub')), unsafe_allow_html=True)
        fig11 = px.histogram(af, x="Tenure_Month", nbins=12, color_discrete_sequence=[C_BLACK])
        fig11.update_traces(marker_line_color=C_WHITE, marker_line_width=1.5)
        fig11.update_xaxes(title_text="Tenure (Months)")
        fig11.update_yaxes(title_text="Count")
        apply_ibcs(fig11, height=300)
        st.plotly_chart(fig11, use_container_width=True)
        st.markdown(card_foot("Distribution of tenure at resignation · 12-bin histogram"), unsafe_allow_html=True)

    with h4:
        st.markdown(card_open(_('c15_title'), _('c15_sub')), unsafe_allow_html=True)
        zs = sf.groupby('Zone_Type').size().reset_index(name='Total')
        zr = af.groupby('Zone_Type').size().reset_index(name='Resigned')
        zt = zs.merge(zr, on='Zone_Type', how='left').fillna(0)
        zt['Turnover_Rate'] = zt['Resigned'] / zt['Total'] * 100
        zt = zt.sort_values('Turnover_Rate', ascending=False)
        mz = zt.iloc[0]['Zone_Type']
        fig15 = go.Figure(go.Bar(
            x=zt['Zone_Type'], y=zt['Turnover_Rate'],
            marker_color=[C_ACCENT if z == mz else C_BLACK for z in zt['Zone_Type']],
            text=zt['Turnover_Rate'].apply(lambda x: f"{x:.1f}%"), textposition='outside',
            textfont=dict(size=10, family='DM Mono')
        ))
        fig15.update_yaxes(title_text="Turnover Rate (%)", ticksuffix="%")
        apply_ibcs(fig15, height=300)
        st.plotly_chart(fig15, use_container_width=True)
        st.markdown(card_foot("Mock zone classification · red = highest turnover zone"), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB D — DATA QUALITY
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-h">D · DATA QUALITY — Raw vs Standardized</div>', unsafe_allow_html=True)
    dq1, dq2 = st.columns(2)
    with dq1:
        st.markdown(card_open("Raw Survey Data", "Input · original records before transformation"), unsafe_allow_html=True)
        st.dataframe(survey_raw.head(10), use_container_width=True)
        st.markdown(card_foot("Source: engagement_survey_raw.csv · no processing applied"), unsafe_allow_html=True)
    with dq2:
        st.markdown(card_open("Standardized Survey Data", "Output · after type conversion, score mapping, null-fill"), unsafe_allow_html=True)
        st.dataframe(survey_df.head(10), use_container_width=True)
        st.markdown(card_foot("Processed: Q2 mapped to 1–5 scale · Happiness mapped Low/Med/High → 1/3/5"), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB E — MACHINE LEARNING
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown(f'<div class="section-h">{_("ml_title")}</div>', unsafe_allow_html=True)

    ml1, ml2 = st.columns([1, 1.5])

    # C16 — Feature importance
    with ml1:
        st.markdown(card_open(_('c16_title'), _('c16_sub')), unsafe_allow_html=True)
        fi       = feat_imp.copy()
        fi_col   = 'Feature_EN' if st.session_state.lang == 'EN' else 'Feature'
        n        = len(fi)
        fi_colors = [C_ACCENT if i == n-1 else C_BLACK if i >= n-2 else C_DARK_GREY if i >= n-4 else C_MID_GREY
                     for i in range(n)]
        fig16 = go.Figure(go.Bar(
            x=fi['Importance'], y=fi[fi_col], orientation='h',
            marker_color=fi_colors,
            text=fi['Importance'].apply(lambda x: f"{x:.3f}"), textposition='outside',
            textfont=dict(size=10, family='DM Mono')
        ))
        fig16.update_xaxes(range=[0, fi['Importance'].max() * 1.3], title_text="Feature Importance")
        apply_ibcs(fig16, height=340, h_bars=True)
        st.plotly_chart(fig16, use_container_width=True)
        st.markdown(card_foot(f"▪ {_('ml_algo')} · 100 trees · balanced class weight"), unsafe_allow_html=True)

        st.info(_("ml_insight"))
        with st.expander(_("ml_algo_desc_title")):
            st.markdown(_("ml_algo_desc"))

    # C17 — Flight risk table
    with ml2:
        st.markdown(card_open(_('c17_title'), _('c17_sub')), unsafe_allow_html=True)
        act = ml_dataset[ml_dataset['Status'] == 'Active'].copy()
        if not act.empty:
            proba = rf_model.predict_proba(act[ml_features])
            act['Flight_Risk'] = proba[:, 1] * 100 if proba.shape[1] == 2 else (100.0 if rf_model.classes_[0] == 1 else 0.0)
            hr = act.sort_values('Flight_Risk', ascending=False).head(10)
            hd = hr[['Staff_ID','Role','Flight_Risk','Avg_Work_Hours','Total_Late','Happy_Score']].copy()
            hd.columns = ['Staff ID','Role','Flight Risk (%)','Work Hours (h)','Late Count','Happiness Score']

            # Build IBCS HTML table
            rows = ""
            for idx, r in hd.iterrows():
                risk_color = C_ACCENT if r['Flight Risk (%)'] > 60 else C_DARK_GREY
                bar_w = int(r['Flight Risk (%)'] * 0.6)
                rows += f"""<tr>
                    <td style="font-family:'DM Sans',sans-serif;color:{C_BLACK};">{r['Staff ID']}</td>
                    <td style="font-family:'DM Sans',sans-serif;">{r['Role']}</td>
                    <td style="color:{risk_color};font-weight:600;">{r['Flight Risk (%)']:.1f}%
                        <span style="display:inline-block;width:{bar_w}px;height:6px;background:{risk_color};margin-left:4px;vertical-align:middle;"></span>
                    </td>
                    <td>{r['Work Hours (h)']:.1f}h</td>
                    <td>{int(r['Late Count'])}</td>
                    <td>{r['Happiness Score']:.1f}</td>
                </tr>"""
            st.markdown(f"""
            <table class="ibcs-table">
                <thead><tr>
                    <th>Staff ID</th><th>Role</th><th>Flight Risk % ▼</th>
                    <th>Work Hrs</th><th>Late ×</th><th>Happy</th>
                </tr></thead>
                <tbody>{rows}</tbody>
            </table>
            """, unsafe_allow_html=True)
            st.markdown(card_foot("Predicted by Random Forest · sorted by flight risk descending · top 10 shown"), unsafe_allow_html=True)
            st.warning(_("ml_action"))
        else:
            st.write(_("ml_no_data"))
            st.markdown("</div>", unsafe_allow_html=True)