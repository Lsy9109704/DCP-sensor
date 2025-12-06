import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. 页面配置
# ---------------------------------------------------------
st.set_page_config(
    page_title="DCP Sensor Analysis", 
    page_icon="🔬", 
    layout="wide"  # 开启宽屏模式，更大气
)

# ---------------------------------------------------------
# 2. 核心 CSS 样式美化 (这是变好看的关键)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* 1. 全局背景色：极淡的灰蓝色，护眼且专业 */
    .stApp {
        background-color: #F8F9FA;
    }

    /* 2. 侧边栏样式：纯白背景 + 右侧阴影 */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        box-shadow: 2px 0 5px rgba(0,0,0,0.05);
        border-right: none;
    }

    /* 3. 优化输入框 (解决您提到的黑色问题) */
    /* 输入框容器 */
    div[data-baseweb="input"] {
        background-color: #FFFFFF !important; /* 纯白背景 */
        border: 1px solid #CED4DA !important; /* 灰色边框 */
        border-radius: 8px !important;       /* 圆角 */
        padding: 5px !important;
    }
    /* 输入时的文字颜色 */
    input[type="number"] {
        color: #2C3E50 !important; /* 深蓝灰色字体，清晰清楚 */
        font-weight: 600 !important; /* 字体加粗 */
        font-size: 16px !important;
    }
    /* 输入框标签 */
    .stNumberInput label {
        color: #555555 !important;
        font-size: 15px !important;
        font-weight: bold;
    }

    /* 4. 按钮样式：渐变蓝色 */
    div.stButton > button {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        width: 100%;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }

    /* 5. 结果卡片样式 */
    .metric-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #4b6cb7; /* 左侧蓝色装饰条 */
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #2C3E50;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 14px;
        color: #888888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* 标题样式 */
    h1 {
        color: #1a2a6c;
        font-family: 'Helvetica Neue', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. 加载模型
# ---------------------------------------------------------
@st.cache_resource
def load_brain():
    try:
        return joblib.load('sensor_brain.pkl')
    except:
        return None

models = load_brain()

# ---------------------------------------------------------
# 4. 侧边栏：输入面板
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2920/2920326.png", width=60)
    st.markdown("### Signal Input Panel")
    st.markdown("Please input the sensor array responses below:")
    
    st.markdown("---")
    
    # 更加美观的输入框
    mpa_val = st.number_input("Array 1 Signal", value=-0.0025, format="%.6f", step=0.0001)
    st.caption("Micro-Pit Array Response")
    
    flu_val = st.number_input("Array 2 Signal", value=0.0000, format="%.6f", step=0.0001)
    st.caption("Planar Fluorescent Response")
    
    st.markdown("---")
    
    run_btn = st.button("Start Analysis 🚀")
    
    st.markdown("<div style='text-align: center; color: #999; margin-top: 20px; font-size: 12px;'>SVM & SVR Powered</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 5. 主界面：展示面板
# ---------------------------------------------------------
st.title("🔬 DCP Sensor Analysis System")
st.markdown("##### Intelligent Identification & Quantification of Chemical Agents")

if run_btn and models:
    scaler = models['scaler']
    clf = models['classifier']
    svr_dcp = models['regressor_dcp']
    svr_hcl = models['regressor_hcl']

    # 数据处理
    X_in = np.array([[mpa_val, flu_val]])
    X_scaled = scaler.transform(X_in)

    # 预测概率
    probs = clf.predict_proba(X_scaled)[0]
    max_prob = np.max(probs)
    pred_idx = np.argmax(probs)

    THRESHOLD = 0.80

    st.write("") # 空一行

    if max_prob < THRESHOLD:
        # === 场景 A：未知物 (红色警告风格) ===
        st.error("⚠️ **Warning: Unknown Substance Detected**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="metric-card" style="border-left-color: #e74c3c;">
                <div class="metric-label">Status</div>
                <div class="metric-value" style="color: #e74c3c;">Unknown</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
             st.markdown(f"""
            <div class="metric-card" style="border-left-color: #e74c3c;">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{max_prob*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.info("The input signal pattern deviates significantly from the calibration curve. Please check for potential interferents.")

    else:
        # === 场景 B：已知物 (绿色/蓝色风格) ===
        species = "DCP" if pred_idx == 0 else "HCl"
        
        # 算浓度
        if species == "DCP":
            log_c = svr_dcp.predict(X_scaled)[0]
            unit = "ppm"
            l_min, l_max = 0.1, 10000
        else:
            log_c = svr_hcl.predict(X_scaled)[0]
            unit = "ppb"
            l_min, l_max = 4, 400000
        
        real_c = 10 ** log_c

        # 判断范围
        status_color = "#27ae60" # 默认绿色
        status_text = "Valid Range"
        
        if real_c < l_min:
            status_text = "Below LOD"
            status_color = "#95a5a6" # 灰色
        elif real_c > l_max:
            status_text = "Saturation"
            status_color = "#e67e22" # 橙色

        # 使用 HTML 卡片布局展示结果 (比 st.metric 更美观)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #4b6cb7;">
                <div class="metric-label">Identified Species</div>
                <div class="metric-value">{species}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: {status_color};">
                <div class="metric-label">Concentration ({unit})</div>
                <div class="metric-value" style="color: {status_color};">{real_c:.3f}</div>
                <div style="font-size:12px; color:{status_color}; font-weight:bold;">{status_text}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #27ae60;">
                <div class="metric-label">AI Confidence</div>
                <div class="metric-value">{max_prob*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # 进度条区域
        st.markdown("### 📊 Linear Range Visualization")
        st.write(f"Standard range for **{species}**: `{l_min}` to `{l_max}` {unit}")
        
        if l_min <= real_c <= l_max:
            norm = (log_c - np.log10(l_min)) / (np.log10(l_max) - np.log10(l_min))
            st.progress(min(max(norm, 0.0), 1.0))
        else:
            if real_c < l_min:
                st.warning(f"Note: The calculated concentration is below the reliable detection limit ({l_min}).")
            else:
                st.warning(f"Note: The signal exceeds the linear calibration range ({l_max}).")

elif run_btn and not models:
    st.error("🚨 Model file missing. Please run 'train_model.py' first.")

else:
    # 初始欢迎界面
    st.info("👈 Please enter the Array signals in the sidebar to start.")
    
    # 装饰性图片或文字
    st.markdown("""
    <div style="background-color: white; padding: 20px; border-radius: 10px; border: 1px solid #eee; margin-top: 20px;">
        <h4 style="color: #4b6cb7; margin-top:0;">System Capabilities:</h4>
        <ul>
            <li><b>Dual-Channel Input:</b> Micro-Pit Array + Planar Film</li>
            <li><b>Algorithm:</b> Support Vector Machine (SVM) + Regression (SVR)</li>
            <li><b>Target Analytes:</b> DCP (0.1-10000 ppm) & HCl (4-400000 ppb)</li>
            <li><b>Safety:</b> Automatic Unknown Substance Rejection</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)