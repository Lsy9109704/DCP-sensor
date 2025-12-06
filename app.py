import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

# ==========================================
# 0. Page Config
# ==========================================
st.set_page_config(page_title="Intelligent Sensor System", page_icon="⚗️", layout="wide")

st.markdown("""
<style>
    .main {background-color: #FFFFFF;}
    h1, h2, h3 {font-family: 'Arial', sans-serif;}
</style>
""", unsafe_allow_html=True)

st.title("⚗️ Intelligent Fluorescent Sensor Array System")
st.markdown("""
**Workflow:**
1. **Classification:** Identify Species (DCP vs HCl).
2. **High-Precision Quantification:** Non-linear (Polynomial) Regression.
""")

# ==========================================
# 1. Core Logic: Polynomial Regression Pipeline
# ==========================================
@st.cache_resource
def build_pipeline():
    # --- A. Dataset ---
    data = {
        "MPA": [-0.00238, -0.00242, -0.00237, -0.00238, -0.0024, -0.0024, -0.00269, -0.00267, -0.00268, -0.0027, -0.00262, -0.00268, -0.00478, -0.0047, -0.00477, -0.00472, -0.00473, -0.00473, -0.00794, -0.00801, -0.00793, -0.00798, -0.00799, -0.00799, -0.01043, -0.01043, -0.01042, -0.01047, -0.01045, -0.01044, -0.02717, -0.02718, -0.02718, -0.02714, -0.0272, -0.02718, -0.00158, -0.00152, -0.00156, -0.00157, -0.0016, -0.00159, -0.0032, -0.00317, -0.00313, -0.00317, -0.00319, -0.00317, -0.00559, -0.0056, -0.00557, -0.0056, -0.00559, -0.00559, -0.00675, -0.00684, -0.00694, -0.00643, -0.00687, -0.00684, -0.01204, -0.01241, -0.01235, -0.01989, -0.0121, -0.01202, -0.02165, -0.0215, -0.0208, -0.0211, -0.02088, -0.0211], 
        "Flu": [0.0]*18 + [-0.00103, -0.00124, -0.00162, -0.00133, -0.00144, -0.0011, -0.00371, -0.00372, -0.00381, -0.00372, -0.00372, -0.00311, -0.00648, -0.00648, -0.00626, -0.00635, -0.00648, -0.00641] + [-0.00414, -0.00413, -0.00512, -0.00453, -0.00499, -0.00422, -0.00514, -0.00513, -0.00512, -0.00553, -0.00588, -0.00522, -0.00604, -0.00681, -0.00691, -0.00612, -0.00632, -0.00632, -0.01269, -0.01275, -0.01267, -0.01245, -0.01303, -0.01279, -0.0221, -0.02241, -0.02237, -0.02242, -0.02299, -0.02301, -0.04369, -0.04188, -0.04581, -0.04172, -0.04128, -0.04193], 
        "Species": ["DCP"]*36 + ["HCl"]*36, 
        "Concentration": [0.1]*6 + [1.0]*6 + [10.0]*6 + [100.0]*6 + [1000.0]*6 + [10000.0]*6 + [4.0]*6 + [40.0]*6 + [400.0]*6 + [4000.0]*6 + [40000.0]*6 + [400000.0]*6
    }
    # Add Water control
    data["MPA"].extend([-0.0006, -0.0005, -0.0007, -0.0006, -0.0006, -0.0005])
    data["Flu"].extend([0.0, 0.0, 0.0, 0.0, 0.00001, -0.00001])
    data["Species"].extend(["Water"] * 6)
    data["Concentration"].extend([0] * 6)

    df = pd.DataFrame(data)
    df['MPA_Abs'] = df['MPA'].abs()
    df['Flu_Abs'] = df['Flu'].abs()

    # --- B. Train Classifier (SVM) ---
    scaler = StandardScaler()
    X_cls = scaler.fit_transform(df[['MPA_Abs', 'Flu_Abs']].values)
    clf = SVC(kernel='linear', C=100, probability=True)
    clf.fit(X_cls, df['Species'].values)

    # --- C. Train Regressor (Polynomial Degree 2) ---
    models_info = {}
    
    for spec in ['DCP', 'HCl']:
        subset = df[df['Species'] == spec].copy()
        
        # Select primary feature
        if spec == 'DCP':
            y_train = subset['MPA_Abs'].values # Signal (Y)
            X_train = np.log10(subset['Concentration'].values).reshape(-1, 1) # Log Conc (X)
            feature_name = 'MPA_Abs'
        else:
            y_train = subset['Flu_Abs'].values
            X_train = np.log10(subset['Concentration'].values).reshape(-1, 1)
            feature_name = 'Flu_Abs'

        # Train Polynomial Model: Signal = f(LogConc)
        # Degree 2 fits the curve much better than linear
        model = make_pipeline(PolynomialFeatures(2), LinearRegression())
        model.fit(X_train, y_train)
        
        models_info[spec] = {
            'model': model,
            'X_train': X_train,
            'y_train': y_train,
            'feature': feature_name
        }
            
    return clf, models_info, scaler, df

clf, models_info, scaler, df = build_pipeline()

# ==========================================
# 2. Sidebar
# ==========================================
with st.sidebar:
    st.header("🎛️ Sensor Input")
    st.info("Input raw values ($F/F_0 - 1$)")
    mpa_val = st.number_input("Sensor 1 (MPA)", value=-0.00250, format="%.5f", step=0.0001)
    flu_val = st.number_input("Sensor 2 (Flu)", value=0.00000, format="%.5f", step=0.0001)
    st.markdown("---")
    analyze_btn = st.button("🚀 Analyze", type="primary")

# ==========================================
# 3. Analysis Logic
# ==========================================
def solve_concentration(model, signal_val):
    # The model predicts Signal from Conc: Signal = a*x^2 + b*x + c
    # We have Signal, we need to solve for x (LogConc).
    # This involves solving the quadratic equation: ax^2 + bx + (c - Signal) = 0
    
    # Extract coefficients
    intercept = model.named_steps['linearregression'].intercept_
    coefs = model.named_steps['linearregression'].coef_
    # coefs[0] is for x^0 (usually 0), coefs[1] is x, coefs[2] is x^2
    
    a = coefs[2]
    b = coefs[1]
    c = intercept - signal_val
    
    # Quadratic formula: x = (-b +/- sqrt(b^2 - 4ac)) / 2a
    delta = b**2 - 4*a*c
    
    if delta < 0:
        return None # No real solution
    
    # Two solutions, pick the physically meaningful one (usually the one within concentration range)
    x1 = (-b + np.sqrt(delta)) / (2*a)
    x2 = (-b - np.sqrt(delta)) / (2*a)
    
    # Heuristic: The relationship is monotonic in our range. 
    # Usually the positive slope branch is what we want.
    # Let's check which one falls in reasonable range (-1 to 6)
    if -2 <= x1 <= 7: return x1
    if -2 <= x2 <= 7: return x2
    return x1 # Default

if analyze_btn:
    if mpa_val > 0.001 or flu_val > 0.001:
        st.error("⚠️ Unknown Interference (Enhancement Detected)")
    else:
        # 1. Classification
        input_abs = np.array([[abs(mpa_val), abs(flu_val)]])
        pred_species = clf.predict(scaler.transform(input_abs))[0]
        
        if pred_species == "Water":
            st.success("✅ Clean / Background")
        else:
            # 2. Quantification (Polynomial Inversion)
            model_data = models_info[pred_species]
            signal_val = abs(mpa_val) if pred_species == 'DCP' else abs(flu_val)
            
            # Solve for Log Concentration
            pred_log = solve_concentration(model_data['model'], signal_val)
            
            if pred_log is None:
                st.warning("Signal out of calibration range.")
                pred_conc = 0
            else:
                pred_conc = 10 ** pred_log

            # Display
            c1, c2 = st.columns(2)
            c1.metric("Species", pred_species)
            c2.metric("Concentration", f"{pred_conc:.2f} ppb")
            
            # 3. Plotting
            st.markdown("---")
            st.subheader(f"📈 Non-linear Calibration Curve")
            
            fig, ax = plt.subplots(figsize=(8, 5))
            
            X_train = model_data['X_train']
            y_train = model_data['y_train']
            
            # Draw Data
            ax.scatter(X_train, y_train, color='blue', alpha=0.6, label='Training Data')
            
            # Draw Curve
            x_range = np.linspace(X_train.min()-0.5, X_train.max()+0.5, 100).reshape(-1, 1)
            y_curve = model_data['model'].predict(x_range)
            ax.plot(x_range, y_curve, 'k-', label='Polynomial Fit (Deg 2)')
            
            # Draw Sample
            if pred_log is not None:
                ax.scatter([pred_log], [signal_val], color='red', marker='*', s=300, 
                           edgecolors='black', label='Your Sample', zorder=10)
            
            # Metrics
            r2 = r2_score(y_train, model_data['model'].predict(X_train))
            ax.text(0.05, 0.9, f"$R^2$ = {r2:.4f}", transform=ax.transAxes, 
                   bbox=dict(facecolor='white', alpha=0.9))
            
            ax.set_xlabel("Log10 (Concentration)", fontweight='bold')
            ax.set_ylabel(f"Signal ({model_data['feature']})", fontweight='bold')
            ax.legend()
            ax.grid(True, linestyle=':', alpha=0.5)
            
            st.pyplot(fig)
