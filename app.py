# ==============================================================================
# STREAMLIT APP DỰ ĐOÁN NGUY CƠ ĐỘT QUỴ - PHIÊN BẢN GIAO DIỆN CHUYÊN NGHIỆP
# Cài đặt thư viện: pip install streamlit pandas numpy plotly scikit-learn joblib
# ==============================================================================

import json
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ------------------------------------------------------------------------------
# 1. CẤU HÌNH TRANG & CUSTOM CSS
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Hệ Thống Dự Đoán Nguy Cơ Đột Quỵ",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0A2463 0%, #1E6091 100%);
        color: white;
        padding: 28px 32px;
        border-radius: 16px;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px -5px rgba(10, 36, 99, 0.2);
    }
    .hero-title {
        font-size: 28px !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        margin-bottom: 6px !important;
    }
    .hero-subtitle {
        font-size: 15px !important;
        color: #E2E8F0 !important;
        margin-bottom: 0px !important;
    }
    
    /* Section Cards */
    .css-1r6slb0, .css-1y4p8pa, div[data-testid="stForm"] {
        background-color: white;
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        border: 1px solid #E2E8F0;
    }
    
    /* Custom Metric Display */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        text-align: center;
    }
    .metric-val {
        font-size: 26px;
        font-weight: 800;
        color: #0A2463;
    }
    .metric-lbl {
        font-size: 13px;
        color: #64748B;
        font-weight: 600;
        margin-top: 4px;
    }
    
    /* Primary Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #1E6091 0%, #184E77 100%);
        color: white !important;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 700;
        font-size: 16px;
        border: none;
        width: 100%;
        box-shadow: 0 4px 12px rgba(30, 96, 145, 0.25);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(30, 96, 145, 0.35);
        background: linear-gradient(135deg, #184E77 0%, #0A2463 100%);
    }

    /* Alert Boxes */
    .alert-high-risk {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        color: #991B1B;
        padding: 16px;
        border-radius: 8px;
        margin-top: 10px;
    }
    .alert-low-risk {
        background-color: #F0FDF4;
        border-left: 5px solid #10B981;
        color: #065F46;
        padding: 16px;
        border-radius: 8px;
        margin-top: 10px;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. HÀM TÌM FILE VÀ LOAD DỮ LIỆU
# ------------------------------------------------------------------------------
def _candidate_paths(filename):
    base = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    return [
        os.path.join("models", filename),
        os.path.join("notebooks", "models", filename),
        os.path.join("..", "models", filename),
        os.path.join("..", "notebooks", "models", filename),
        os.path.join(base, "models", filename),
        os.path.join(base, "notebooks", "models", filename),
        os.path.join(base, "..", "models", filename),
        os.path.join(base, filename),
        filename
    ]

def _load_json(filename):
    for path in _candidate_paths(filename):
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return None

def _load_csv(filename):
    for path in _candidate_paths(filename):
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception:
                pass
    return None

def _load_first_available_csv(filenames):
    for fname in filenames:
        df = _load_csv(fname)
        if df is not None and not df.empty:
            return df
    return None

@st.cache_resource
def load_stroke_model():
    model, model_path = None, None
    for path in _candidate_paths("stroke_model.pkl"):
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                model, model_path = joblib.load(path), path
                break
            except Exception as e:
                return None, None, None, f"Lỗi khi tải mô hình: {e}"

    threshold_data = _load_json("threshold.json") or {}
    threshold = float(threshold_data.get("threshold", 0.5))
    target_recall = float(threshold_data.get("target_recall", 0.80))
    best_model_name = threshold_data.get("model", "Unknown Model")
    target_column = threshold_data.get("target_column", "stroke")

    summary_data = _load_json("summary.json") or {}

    return model, model_path, {
        "threshold": threshold,
        "target_recall": target_recall,
        "best_model_name": best_model_name,
        "target_column": target_column,
        "summary": summary_data,
    }, None

model, loaded_path, model_meta, load_error = load_stroke_model()

if model_meta is None:
    model_meta = {
        "threshold": 0.5,
        "target_recall": 0.80,
        "best_model_name": "Unknown",
        "target_column": "stroke",
        "summary": {},
    }

threshold = model_meta["threshold"]
TARGET_COL = model_meta["target_column"]

df_comparison = _load_first_available_csv([
    "model_comparison.csv",
    "evaluation_summary.csv",
    "model_comparison_smote.csv",
])

df_sample = _load_first_available_csv([
    "eda_sample.csv",
    "stroke_clean.csv",
    "stroke_risk_dataset_v2.csv"
])

# ------------------------------------------------------------------------------
# 3. TRÍCH XUẤT FEATURE IMPORTANCE TỪ MODEL
# ------------------------------------------------------------------------------
@st.cache_data
def extract_feature_importance(model_path):
    if model_path is None:
        return None

    try:
        pipeline = joblib.load(model_path)
        if hasattr(pipeline, "calibrated_classifiers_"):
            inner_pipe = pipeline.calibrated_classifiers_[0].estimator
        else:
            inner_pipe = pipeline

        if hasattr(inner_pipe, "named_steps"):
            pre = inner_pipe.named_steps.get("preprocessor")
            clf = inner_pipe.named_steps.get("classifier")

            if pre is not None:
                try:
                    raw_names = pre.get_feature_names_out()
                    feature_names = [n.split("__", 1)[1] if "__" in n else n for n in raw_names]
                except Exception:
                    feature_names = None
            else:
                feature_names = None

            if clf is None:
                return None

            if hasattr(clf, "feature_importances_"):
                importances = clf.feature_importances_
            elif hasattr(clf, "coef_"):
                importances = np.abs(clf.coef_).flatten()
            else:
                return None

            if feature_names is None or len(feature_names) != len(importances):
                feature_names = [f"Đặc trưng {i+1}" for i in range(len(importances))]

            df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances,
            }).sort_values("Importance", ascending=False).reset_index(drop=True)

            name_map = {
                "age": "Tuổi",
                "hypertension": "Tăng huyết áp",
                "heart_disease": "Bệnh tim mạch",
                "avg_glucose_level": "Mức Glucose trung bình",
                "bmi": "Chỉ số BMI",
                "gender_Male": "Giới tính (Nam)",
                "gender_Female": "Giới tính (Nữ)",
                "ever_married_Yes": "Đã kết hôn",
                "ever_married_No": "Chưa kết hôn",
                "smoking_status_smokes": "Hút thuốc",
                "smoking_status_never smoked": "Không hút thuốc",
                "smoking_status_formerly smoked": "Từng hút thuốc",
                "smoking_status_Unknown": "Không rõ hút thuốc",
                "work_type_Private": "Làm tư nhân",
                "work_type_Self-employed": "Tự kinh doanh",
                "work_type_Govt_job": "Nhà nước",
                "work_type_children": "Trẻ em",
                "work_type_Never_worked": "Chưa làm việc",
                "Residence_type_Urban": "Thành thị",
                "Residence_type_Rural": "Nông thôn",
            }
            df["Feature"] = df["Feature"].map(lambda x: name_map.get(x, x))
            return df.head(10)
    except Exception:
        return None
    return None

feature_importance_data = extract_feature_importance(loaded_path)

# ------------------------------------------------------------------------------
# 4. SIDEBAR NAVIGATION
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
            <div style="font-size: 38px; background: #0A2463; padding: 8px 12px; border-radius: 12px; color: white;">🧠</div>
            <div>
                <h3 style="margin: 0; color: #0A2463; font-weight: 800; font-size: 20px;">Stroke AI</h3>
                <p style="margin: 0; color: #64748B; font-size: 12px; font-weight: 500;">Hệ Thống Chẩn Đoán Nguy Cơ</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    menu_items = [
        "🏠 Trang chủ",
        "🔮 Dự đoán nguy cơ",
        "📊 Khám phá Dataset",
        "📈 Phân tích dữ liệu",
        "⚙️ Thông tin Mô hình",
        "✅ Đánh giá & So sánh",
    ]
    selected_menu = st.radio(
        "ĐIỀU HƯỚNG TÍNH NĂNG", menu_items, index=1
    )

    st.markdown("---")
    st.markdown("### ⚙️ Trạng thái Hệ thống")
    if model is not None:
        st.success("✅ Mô hình đã sẵn sàng")
        st.info(f"🎯 Ngưỡng cắt (Threshold): **{threshold * 100:.2f}%**")
        st.caption(f"🤖 Thuật toán: **{model_meta['best_model_name']}**")
        st.caption(f"🏷️ Biến mục tiêu: **{TARGET_COL}**")
    else:
        st.error("❌ Chưa tìm thấy file mô hình (.pkl)")

    st.markdown("---")
    st.caption("⚠️ *Lưu ý: Kết quả dự đoán mang tính chất tham khảo y học, không thay thế chẩn đoán chuyên môn từ bác sĩ.*")

# ------------------------------------------------------------------------------
# 5. NỘI DUNG CHÍNH THEO TRANG
# ------------------------------------------------------------------------------

# ==============================================================================
# TRANG 1: TRANG CHỦ
# ==============================================================================
if selected_menu == "🏠 Trang chủ":
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">🧠 Hệ Thống Dự Đoán Nguy Cơ Đột Quỵ</div>
            <div class="hero-subtitle">Giải pháp hỗ trợ tầm soát sớm nguy cơ đột quỵ ứng dụng các thuật toán Học Máy tiên tiến.</div>
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-val">90%+</div>
                <div class="metric-lbl">Độ nhạy (Recall) Tối ưu</div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{threshold*100:.1f}%</div>
                <div class="metric-lbl">Ngưỡng Phân loại (Threshold)</div>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{model_meta['best_model_name']}</div>
                <div class="metric-lbl">Mô hình Tối ưu Nhất</div>
            </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-val">10+</div>
                <div class="metric-lbl">Chỉ số Lâm sàng Đánh giá</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1.2, 1], gap="large")
    with col_left:
        with st.container(border=True):
            st.markdown("### 📌 Hướng dẫn sử dụng hệ thống")
            st.markdown("""
            1. **Bước 1**: Chọn mục **🔮 Dự đoán nguy cơ** tại danh mục điều hướng bên trái.
            2. **Bước 2**: Nhập chính xác các thông tin chỉ số sinh hiệu và lâm sàng của bệnh nhân (Tuổi, Huyết áp, Bệnh tim, BMI, Glucose,...).
            3. **Bước 3**: Nhấn nút **🔮 Dự đoán nguy cơ** để nhận phân tích xác suất nguy cơ.
            4. **Bước 4**: Xem biểu đồ đồng hồ nguy cơ, khuyến nghị y khoa cá nhân hóa và các yếu tố ảnh hưởng chính.
            """)
    with col_right:
        with st.container(border=True):
            st.markdown("### 🔍 Yếu tố nguy cơ hàng đầu")
            st.markdown("""
            - 📈 **Tuổi tác**: Nguy cơ đột quỵ gia tăng đáng kể sau tuổi 55.
            - 🩸 **Tăng huyết áp**: Nguyên nhân dẫn hàng đầu gây tổn thương mạch máu não.
            - 🍬 **Đường huyết cao (Glucose)**: Ảnh hưởng tiêu cực đến hệ tuần hoàn.
            - 🫀 **Bệnh tim mạch**: Gia tăng nguy cơ hình thành cục máu đông.
            - ⚖️ **Chỉ số BMI**: Béo phì làm gia tăng áp lực lên hệ tim mạch.
            """)

# ==============================================================================
# TRANG 2: DỰ ĐOÁN NGUY CƠ
# ==============================================================================
elif selected_menu == "🔮 Dự đoán nguy cơ":
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">🔮 Dự Đoán Nguy Cơ Đột Quỵ</div>
            <div class="hero-subtitle">Nhập các tham số sức khỏe lâm sàng để ước tính xác suất nguy cơ đột quỵ của bệnh nhân.</div>
        </div>
    """, unsafe_allow_html=True)

    if "prob_percent" not in st.session_state:
        st.session_state["prob_percent"] = 0.0
        st.session_state["is_high_risk"] = False
        st.session_state["has_predicted"] = False

    col_input, col_output = st.columns([1.1, 1.4], gap="large")

    # CỘT TRÁI: FORM NHẬP LIỆU
    with col_input:
        with st.container(border=True):
            st.markdown("### 📋 Thông tin Lâm sàng Bệnh nhân")

            age = st.slider("Tuổi bệnh nhân", min_value=1.0, max_value=100.0, value=52.0, step=1.0)

            c1, c2 = st.columns(2)
            with c1:
                gender = st.radio("Giới tính", ["Nam", "Nữ"], horizontal=True)
            with c2:
                marital_status = st.radio("Tình trạng hôn nhân", ["Độc thân", "Đã kết hôn"], horizontal=True)

            c3, c4 = st.columns(2)
            with c3:
                hypertension = st.toggle("Tăng huyết áp", value=False)
            with c4:
                heart_disease = st.toggle("Bệnh tim mạch", value=False)

            st.markdown("##### 📏 Chỉ số khối cơ thể (BMI)")
            c5, c6 = st.columns(2)
            with c5:
                weight = st.number_input("Cân nặng (kg)", min_value=30.0, max_value=200.0, value=68.0, step=0.5)
            with c6:
                height_cm = st.number_input("Chiều cao (cm)", min_value=100.0, max_value=220.0, value=165.0, step=1.0)

            bmi = weight / ((height_cm / 100) ** 2)

            if bmi < 18.5:
                bmi_status, bmi_color = "Gầy (Thiếu cân)", "#F59E0B"
            elif bmi < 25.0:
                bmi_status, bmi_color = "Bình thường (Chuẩn)", "#10B981"
            elif bmi < 30.0:
                bmi_status, bmi_color = "Thừa cân", "#F59E0B"
            else:
                bmi_status, bmi_color = "Béo phì", "#EF4444"

            st.markdown(
                f"""
                <div style="background-color: {bmi_color}15; padding: 10px 14px;
                            border-radius: 8px; border-left: 4px solid {bmi_color}; margin-bottom: 15px;">
                    <span style="font-size: 14px; color: #1E293B; font-weight: 600;">
                        Chỉ số BMI: <b>{bmi:.2f}</b> — <span style="color: {bmi_color};">{bmi_status}</span>
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            avg_glucose_level = st.number_input(
                "Mức Glucose trung bình (mg/dL)",
                min_value=50.0,
                max_value=300.0,
                value=106.0,
                step=1.0,
                help="Mức đường huyết bình thường thường nằm trong khoảng 70 - 140 mg/dL."
            )

            smoking_status = st.selectbox(
                "Tình trạng hút thuốc",
                ["never smoked", "formerly smoked", "smokes", "Unknown"],
                format_func=lambda x: {
                    "never smoked": "Chưa bao giờ hút thuốc",
                    "formerly smoked": "Đã từng hút (Đã bỏ)",
                    "smokes": "Hiện đang hút thuốc",
                    "Unknown": "Không xác định"
                }.get(x, x)
            )

            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("🔮 Bắt đầu Dự đoán Nguy cơ", use_container_width=True)

    # CỘT PHẢI: KẾT QUẢ VÀ HIỂN THỊ
    with col_output:
        if predict_btn:
            if model is None:
                st.error("❌ Không thể chạy dự đoán do chưa tải được file mô hình .pkl.")
            else:
                gender_en = "Male" if gender == "Nam" else "Female"
                ever_married_en = "Yes" if marital_status == "Đã kết hôn" else "No"

                if age >= 18:
                    work_type = "Private"
                elif age < 16:
                    work_type = "children"
                else:
                    work_type = "Never_worked"
                residence_type = "Urban"

                input_data = pd.DataFrame([{
                    "gender": gender_en,
                    "age": age,
                    "hypertension": 1 if hypertension else 0,
                    "heart_disease": 1 if heart_disease else 0,
                    "ever_married": ever_married_en,
                    "work_type": work_type,
                    "Residence_type": residence_type,
                    "avg_glucose_level": avg_glucose_level,
                    "bmi": bmi,
                    "smoking_status": smoking_status,
                }])

                try:
                    probability = float(model.predict_proba(input_data)[0][1])
                    prob_percent = probability * 100
                    is_high_risk = probability >= threshold

                    st.session_state["prob_percent"] = prob_percent
                    st.session_state["is_high_risk"] = is_high_risk
                    st.session_state["has_predicted"] = True
                except Exception as e:
                    st.error(f"❌ Lỗi dự đoán từ mô hình: {e}")

        prob_percent = st.session_state["prob_percent"]
        is_high_risk = st.session_state["is_high_risk"]
        has_predicted = st.session_state["has_predicted"]

        # HÀNG 1: ĐỒNG HỒ GAUGE VÀ THẺ KẾT LUẬN
        c_gauge, c_warning = st.columns([1, 1.1])

        with c_gauge:
            with st.container(border=True):
                st.markdown("#### ⏱️ Đồng hồ Nguy cơ")
                bar_color = "#10B981" if not is_high_risk else "#EF4444"

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_percent,
                    number={
                        "suffix": "%",
                        "font": {
                            "size": 38,
                            "color": "#0A2463",
                            "family": "Inter, sans-serif",
                        },
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100],
                            "tickwidth": 1,
                            "tickcolor": "#94A3B8",
                        },
                        "bar": {"color": bar_color, "thickness": 0.35},
                        "bgcolor": "white",
                        "borderwidth": 1,
                        "bordercolor": "#E2E8F0",
                        "steps": [
                            {"range": [0, threshold * 100], "color": "#ECFDF5"},
                            {"range": [threshold * 100, 100], "color": "#FEF2F2"},
                        ],
                    },
                ))
                fig_gauge.update_layout(
                    height=200, margin=dict(l=10, r=10, t=10, b=10)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

        with c_warning:
            with st.container(border=True):
                st.markdown("#### 🏥 Đánh giá & Khuyến nghị")
                st.markdown("<br>", unsafe_allow_html=True)
                if has_predicted:
                    if is_high_risk:
                        st.markdown(
                            f"""
                            <div class="alert-high-risk">
                                <strong style="font-size: 16px;">⚠️ CẢNH BÁO: NGUY CƠ CAO ({prob_percent:.2f}%)</strong><br>
                                <span style="font-size: 13px;">Mức xác suất vượt quá ngưỡng an toàn ({threshold*100:.1f}%). Bệnh nhân nên tham vấn bác sĩ chuyên khoa tim mạch / thần kinh để làm thêm kiểm tra chuyên sâu.</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="alert-low-risk">
                                <strong style="font-size: 16px;">✅ NGUY CƠ THẤP ({prob_percent:.2f}%)</strong><br>
                                <span style="font-size: 13px;">Xác suất nằm trong vùng an toàn. Khuyến nghị duy trì chế độ ăn uống lành mạnh, tập thể dục thường xuyên và kiểm tra sức khỏe định kỳ.</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("👈 Nhập thông tin bên trái và nhấn **'Bắt đầu Dự đoán'** để nhận đánh giá chi tiết.")

        # HÀNG 2: KHUYẾN NGHỊ SỨC KHỎE CÁ NHÂN HÓA
        if has_predicted:
            with st.container(border=True):
                st.markdown("#### 💡 Khuyên dùng cho Bệnh nhân")
                recs = []
                if hypertension:
                    recs.append("• **Kiểm soát huyết áp**: Cần theo dõi huyết áp hàng ngày, giảm lượng muối dưới 5g/ngày.")
                if heart_disease:
                    recs.append("• **Theo dõi tim mạch**: Định kỳ đo điện tâm đồ (ECG) và tái khám đúng hẹn.")
                if avg_glucose_level > 140:
                    recs.append("• **Kiểm soát đường huyết**: Mức glucose cao (>140 mg/dL), cần tham vấn xét nghiệm HbA1c.")
                if bmi >= 25:
                    recs.append("• **Tối ưu cân nặng**: Tập luyện thể thao ít nhất 150 phút/tuần để cải thiện BMI.")
                if smoking_status in ["smokes", "formerly smoked"]:
                    recs.append("• **Cai thuốc lá**: Hạn chế triệt để chất kích thích để bảo vệ thành mạch máu.")
                if not recs:
                    recs.append("• Duy trì thói quen sinh hoạt khoa học, uống đủ 2 lít nước mỗi ngày và ngủ đủ 7-8 tiếng.")

                for r in recs:
                    st.markdown(r)

        # HÀNG 3: FEATURE IMPORTANCE
        with st.container(border=True):
            st.markdown("#### 📊 Tầm Quan Trọng Của Các Đặc Trưng (Top Features)")
            if feature_importance_data is not None and not feature_importance_data.empty:
                fig_feat = px.bar(
                    feature_importance_data,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale="Blues",
                )
                fig_feat.update_layout(
                    height=280,
                    margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=False,
                    coloraxis_showscale=False,
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_feat, use_container_width=True)
            else:
                st.info("ℹ️ Không thể trích xuất Feature Importance từ cấu trúc mô hình hiện tại.")

# ==============================================================================
# TRANG 3: KHÁM PHÁ DATASET
# ==============================================================================
elif selected_menu == "📊 Khám phá Dataset":
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">📊 Khám Phá Tập Dữ Liệu</div>
            <div class="hero-subtitle">Xem chi tiết dữ liệu mẫu và các thông số thống kê mô tả của Dataset huấn luyện.</div>
        </div>
    """, unsafe_allow_html=True)

    if df_sample is not None and not df_sample.empty:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Tổng số dòng (Rows)", df_sample.shape[0])
        with c2:
            st.metric("Tổng số cột (Columns)", df_sample.shape[1])
        with c3:
            st.metric("Biến mục tiêu", TARGET_COL)

        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("### 🔍 Dữ liệu Mẫu (Sample Rows)")
            n_rows = st.slider("Số lượng dòng hiển thị", 5, 50, 10)
            st.dataframe(df_sample.head(n_rows), use_container_width=True)

        with st.container(border=True):
            st.markdown("### 📈 Thống kê Mô tả (Descriptive Statistics)")
            st.dataframe(df_sample.describe().T, use_container_width=True)
    else:
        st.warning("⚠️ Chưa tìm thấy file dữ liệu mẫu (`eda_sample.csv` hoặc dataset tương đương).")

# ==============================================================================
# TRANG 4: PHÂN TÍCH DỮ LIỆU (EDA)
# ==============================================================================
elif selected_menu == "📈 Phân tích dữ liệu":
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">📈 Phân Tích Dữ Liệu Trực Quan (EDA)</div>
            <div class="hero-subtitle">Phân tích mối quan hệ giữa các chỉ số sinh hiệu và nguy cơ đột quỵ.</div>
        </div>
    """, unsafe_allow_html=True)

    if df_sample is not None and not df_sample.empty:
        col_target = TARGET_COL if TARGET_COL in df_sample.columns else df_sample.columns[-1]

        c_eda1, c_eda2 = st.columns(2, gap="large")
        with c_eda1:
            with st.container(border=True):
                st.markdown("### ⚖️ Phân Bố Nhãn Mục Tiêu (Class Imbalance)")
                target_counts = df_sample[col_target].value_counts().reset_index()
                target_counts.columns = ["Class", "Count"]
                target_counts["Class_Label"] = target_counts["Class"].map({0: "Không nguy cơ (0)", 1: "Có nguy cơ (1)"})

                fig_pie = px.pie(
                    target_counts,
                    names="Class_Label",
                    values="Count",
                    color="Class_Label",
                    color_discrete_map={"Không nguy cơ (0)": "#1E6091", "Có nguy cơ (1)": "#E76F51"},
                    hole=0.4
                )
                fig_pie.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_pie, use_container_width=True)

        with c_eda2:
            with st.container(border=True):
                st.markdown("### 🎂 Phân Bố Tuổi Theo Nhóm Nguy Cơ")
                if "age" in df_sample.columns:
                    fig_age = px.histogram(
                        df_sample,
                        x="age",
                        color=col_target,
                        barmode="overlay",
                        color_discrete_map={0: "#1E6091", 1: "#E76F51"},
                        opacity=0.7
                    )
                    fig_age.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig_age, use_container_width=True)
                else:
                    st.info("Không có cột 'age' trong dataset.")

        with st.container(border=True):
            st.markdown("### 🌡️ Ma Trận Tương Quan (Correlation Heatmap)")
            num_df = df_sample.select_dtypes(include=[np.number])
            if not num_df.empty:
                corr = num_df.corr()
                fig_corr = px.imshow(
                    corr,
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    aspect="auto"
                )
                fig_corr.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("⚠️ Cần có file dataset để hiển thị biểu đồ EDA.")

# ==============================================================================
# TRANG 5: THÔNG TIN MÔ HÌNH
# ==============================================================================
elif selected_menu == "⚙️ Thông tin Mô hình":
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">⚙️ Cấu Hình Mô Hình Machine Learning</div>
            <div class="hero-subtitle">Chi tiết về mô hình đang hoạt động, đường dẫn lưu và các tham số tối ưu.</div>
        </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        with st.container(border=True):
            st.markdown("### 🤖 Thông số Mô hình Tối ưu")
            st.write(f"- **Tên mô hình chọn**: `{model_meta['best_model_name']}`")
            st.write(f"- **Ngưỡng quyết định (Threshold)**: `{threshold:.4f}` ({threshold*100:.2f}%)")
            st.write(f"- **Mục tiêu Recall đặt ra**: `{model_meta['target_recall']*100:.1f}%`")
            st.write(f"- **Biến dự đoán (Target)**: `{TARGET_COL}`")
            st.write(f"- **Đường dẫn File**: `{loaded_path or 'N/A'}`")

    with c2:
        with st.container(border=True):
            st.markdown("### 📜 Chi tiết Cấu hình JSON (`threshold.json`)")
            st.json(model_meta)

# ==============================================================================
# TRANG 6: ĐÁNH GIÁ VÀ SO SÁNH
# ==============================================================================
elif selected_menu == "✅ Đánh giá & So sánh":
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">✅ So Sánh & Đánh Giá Hiệu Năng Mô Hình</div>
            <div class="hero-subtitle">So sánh kết quả thực nghiệm giữa các thuật toán Machine Learning khác nhau.</div>
        </div>
    """, unsafe_allow_html=True)

    if df_comparison is not None and not df_comparison.empty:
        with st.container(border=True):
            st.markdown("### 📊 Đồ thị So sánh Chỉ số Hiệu năng")

            possible_metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
            metrics = [m for m in possible_metrics if m in df_comparison.columns]
            model_col = "Model" if "Model" in df_comparison.columns else df_comparison.columns[0]

            selected_metric = st.selectbox("Chọn chỉ số muốn so sánh", metrics if metrics else df_comparison.columns[1:])

            df_sorted = df_comparison.sort_values(by=selected_metric, ascending=False)

            fig_bar = px.bar(
                df_sorted,
                x=model_col,
                y=selected_metric,
                color=selected_metric,
                color_continuous_scale="Viridis",
                text_auto=".3f"
            )
            fig_bar.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_bar, use_container_width=True)

        with st.container(border=True):
            st.markdown("### 📋 Bảng So Sánh Chi Tiết")
            if metrics:
                st.dataframe(
                    df_sorted.style.highlight_max(subset=metrics, color="#D1FAE5").format(precision=4),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.dataframe(df_sorted, use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ Chưa tìm thấy file bảng kết quả so sánh mô hình (`model_comparison.csv` hoặc tương đương).")