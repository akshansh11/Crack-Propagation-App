import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle, Polygon
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Set page config
st.set_page_config(page_title="Crack Propagation Analysis", layout="wide")

# Custom CSS for dark theme similar to the images
st.markdown("""
<style>
    .main {
        background-color: #2E2E2E;
        color: white;
    }
    .stApp {
        background-color: #2E2E2E;
    }
    .sidebar .sidebar-content {
        background-color: #3E3E3E;
    }
    h1, h2, h3 {
        color: #FFFFFF;
    }
    .stSelectbox > div > div {
        background-color: #4E4E4E;
        color: white;
    }
    .stSlider > div > div > div {
        background-color: #4E4E4E;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔬 Crack Propagation Analysis")
st.markdown("Advanced fracture mechanics simulation with stress intensity factor calculations")

# Sidebar for input parameters
with st.sidebar:
    st.header("Material Properties")
    
    # Material selection
    material = st.selectbox("Material Type", 
                           ["Steel (AISI 4140)", "Aluminum (7075-T6)", "Titanium (Ti-6Al-4V)", "Custom"])
    
    if material == "Custom":
        E = st.number_input("Young's Modulus (GPa)", value=200.0, min_value=1.0, max_value=1000.0)
        nu = st.number_input("Poisson's Ratio", value=0.3, min_value=0.1, max_value=0.5)
        yield_strength = st.number_input("Yield Strength (MPa)", value=800.0, min_value=100.0, max_value=3000.0)
        K_IC = st.number_input("Fracture Toughness K_IC (MPa√m)", value=50.0, min_value=10.0, max_value=200.0)
    else:
        # Predefined material properties
        materials_db = {
            "Steel (AISI 4140)": {"E": 200, "nu": 0.29, "yield": 800, "K_IC": 50},
            "Aluminum (7075-T6)": {"E": 72, "nu": 0.33, "yield": 500, "K_IC": 35},
            "Titanium (Ti-6Al-4V)": {"E": 110, "nu": 0.34, "yield": 900, "K_IC": 75}
        }
        props = materials_db[material]
        E, nu, yield_strength, K_IC = props["E"], props["nu"], props["yield"], props["K_IC"]
    
    st.header("Geometry")
    crack_type = st.selectbox("Crack Configuration", 
                             ["Center Crack", "Edge Crack", "Corner Crack", "Penny-shaped Crack"])
    
    # Geometry parameters
    if crack_type in ["Center Crack", "Edge Crack"]:
        width = st.slider("Specimen Width (mm)", 10, 200, 100)
        height = st.slider("Specimen Height (mm)", 10, 200, 100)
        crack_length = st.slider("Crack Length (mm)", 1, min(width, height)//2, 20)
    else:
        width = st.slider("Specimen Width (mm)", 10, 200, 100)
        height = st.slider("Specimen Height (mm)", 10, 200, 100)
        crack_length = st.slider("Crack Length (mm)", 1, min(width, height)//4, 15)
    
    st.header("Loading Conditions")
    stress_type = st.selectbox("Loading Type", ["Tension", "Bending", "Mixed Mode"])
    applied_stress = st.slider("Applied Stress (MPa)", 10, yield_strength, 200)
    
    if stress_type == "Mixed Mode":
        mode_I_ratio = st.slider("Mode I Ratio", 0.0, 1.0, 0.7)
        mode_II_ratio = 1 - mode_I_ratio
    
    st.header("Analysis Parameters")
    num_cycles = st.slider("Number of Load Cycles", 1, 10000, 1000)
    delta_K_threshold = st.slider("ΔK Threshold (MPa√m)", 1.0, 10.0, 3.0)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Calculate stress intensity factor
    def calculate_stress_intensity_factor(crack_type, applied_stress, crack_length, width, height):
        """Calculate stress intensity factor based on crack geometry"""
        a = crack_length / 1000  # Convert to meters
        
        if crack_type == "Center Crack":
            # For center crack in infinite plate
            Y = np.sqrt(np.pi * a) * (1 + 0.128 * (a / (width/1000)) - 0.288 * (a / (width/1000))**2)
            K_I = applied_stress * 1e6 * Y
        elif crack_type == "Edge Crack":
            # For edge crack
            alpha = a / (width/1000)
            Y = 1.12 - 0.231 * alpha + 10.55 * alpha**2 - 21.72 * alpha**3 + 30.39 * alpha**4
            K_I = applied_stress * 1e6 * Y * np.sqrt(np.pi * a)
        elif crack_type == "Corner Crack":
            # For corner crack
            alpha = a / (width/1000)
            Y = 1.1 + 0.35 * alpha
            K_I = applied_stress * 1e6 * Y * np.sqrt(np.pi * a)
        else:  # Penny-shaped crack
            # For penny-shaped crack
            Y = 2 / np.pi
            K_I = applied_stress * 1e6 * Y * np.sqrt(np.pi * a)
        
        return K_I / 1e6  # Convert back to MPa√m

    # Calculate Paris law parameters
    def paris_law_crack_growth(K_I, delta_K_threshold, num_cycles):
        """Calculate crack growth using Paris law"""
        # Typical Paris law constants for steel
        C = 3e-12  # Material constant
        m = 3.0    # Paris exponent
        
        delta_K = K_I  # Assuming R = 0 (fully reversed loading)
        
        if delta_K > delta_K_threshold:
            da_dN = C * (delta_K * 1e6)**m  # Crack growth rate m/cycle
            crack_growth = da_dN * num_cycles * 1000  # Convert to mm
        else:
            crack_growth = 0
            
        return crack_growth, da_dN if delta_K > delta_K_threshold else 0

    # Calculate current values
    K_I = calculate_stress_intensity_factor(crack_type, applied_stress, crack_length, width, height)
    crack_growth, da_dN = paris_law_crack_growth(K_I, delta_K_threshold, num_cycles)
    
    # Safety factor
    safety_factor = K_IC / K_I if K_I > 0 else float('inf')
    
    # Create visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Crack Geometry", "Stress Intensity Factor", "Crack Growth Rate", "Phase Field Damage"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # Crack geometry plot
    x_crack = np.linspace(-crack_length/2, crack_length/2, 100)
    y_crack = np.zeros_like(x_crack)
    
    fig.add_trace(
        go.Scatter(x=[-width/2, width/2, width/2, -width/2, -width/2],
                   y=[-height/2, -height/2, height/2, height/2, -height/2],
                   mode='lines', name='Specimen', line=dict(color='white', width=2)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=x_crack, y=y_crack, mode='lines', name='Crack',
                   line=dict(color='red', width=4)),
        row=1, col=1
    )
    
    # Stress intensity factor plot
    crack_lengths = np.linspace(1, min(width, height)//2, 50)
    K_values = [calculate_stress_intensity_factor(crack_type, applied_stress, a, width, height) 
                for a in crack_lengths]
    
    fig.add_trace(
        go.Scatter(x=crack_lengths, y=K_values, mode='lines+markers', name='K_I',
                   line=dict(color='cyan', width=2)),
        row=1, col=2
    )
    
    fig.add_hline(y=K_IC, line_dash="dash", line_color="red", 
                  annotation_text=f"K_IC = {K_IC} MPa√m", row=1, col=2)
    
    # Crack growth rate plot
    cycles = np.linspace(1, num_cycles, 100)
    growth_rates = [paris_law_crack_growth(K_I, delta_K_threshold, n)[0] for n in cycles]
    
    fig.add_trace(
        go.Scatter(x=cycles, y=growth_rates, mode='lines', name='Crack Growth',
                   line=dict(color='yellow', width=2)),
        row=2, col=1
    )
    
    # Phase field damage visualization (simplified)
    x_field = np.linspace(-width/2, width/2, 50)
    y_field = np.linspace(-height/2, height/2, 50)
    X, Y = np.meshgrid(x_field, y_field)
    
    # Simplified damage field (higher damage near crack)
    damage = np.exp(-((X**2 + Y**2) / (crack_length**2 / 4)))
    
    fig.add_trace(
        go.Heatmap(z=damage, x=x_field, y=y_field, colorscale='Viridis',
                   showscale=False, name='Damage Field'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title="Crack Propagation Analysis",
        plot_bgcolor='#2E2E2E',
        paper_bgcolor='#2E2E2E',
        font=dict(color='white'),
        height=800
    )
    
    # Update axes
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(gridcolor='#4E4E4E', row=i, col=j)
            fig.update_yaxes(gridcolor='#4E4E4E', row=i, col=j)
    
    fig.update_xaxes(title_text="Position (mm)", row=1, col=1)
    fig.update_yaxes(title_text="Position (mm)", row=1, col=1)
    fig.update_xaxes(title_text="Crack Length (mm)", row=1, col=2)
    fig.update_yaxes(title_text="K_I (MPa√m)", row=1, col=2)
    fig.update_xaxes(title_text="Cycles", row=2, col=1)
    fig.update_yaxes(title_text="Crack Growth (mm)", row=2, col=1)
    fig.update_xaxes(title_text="X (mm)", row=2, col=2)
    fig.update_yaxes(title_text="Y (mm)", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.header("Results")
    
    # Display key results
    st.metric("Stress Intensity Factor", f"{K_I:.2f} MPa√m")
    st.metric("Fracture Toughness", f"{K_IC:.2f} MPa√m")
    st.metric("Safety Factor", f"{safety_factor:.2f}")
    
    if safety_factor < 1.5:
        st.error("⚠️ Critical: Safety factor is low!")
    elif safety_factor < 2.5:
        st.warning("⚠️ Caution: Safety factor is moderate")
    else:
        st.success("✅ Safe: Adequate safety factor")
    
    st.metric("Crack Growth Rate", f"{da_dN:.2e} m/cycle" if da_dN > 0 else "Below threshold")
    st.metric("Total Crack Growth", f"{crack_growth:.3f} mm")
    
    # Fatigue life estimation
    if da_dN > 0:
        remaining_life = (K_IC - K_I) / (da_dN * 1000 * applied_stress)  # Simplified
        st.metric("Estimated Fatigue Life", f"{remaining_life:.0f} cycles")
    
    st.header("Material Properties")
    st.write(f"**Young's Modulus:** {E} GPa")
    st.write(f"**Poisson's Ratio:** {nu}")
    st.write(f"**Yield Strength:** {yield_strength} MPa")
    st.write(f"**Fracture Toughness:** {K_IC} MPa√m")
    
    st.header("Geometry")
    st.write(f"**Crack Type:** {crack_type}")
    st.write(f"**Dimensions:** {width} × {height} mm")
    st.write(f"**Crack Length:** {crack_length} mm")
    
    st.header("Loading")
    st.write(f"**Applied Stress:** {applied_stress} MPa")
    st.write(f"**Loading Type:** {stress_type}")

# Additional analysis section
st.header("Advanced Analysis")

tab1, tab2, tab3 = st.tabs(["Parametric Study", "Fatigue Analysis", "Failure Assessment"])

with tab1:
    st.subheader("Parametric Study")
    
    # Create parametric study
    param_study = st.selectbox("Study Parameter", 
                              ["Crack Length", "Applied Stress", "Material Properties"])
    
    if param_study == "Crack Length":
        crack_range = np.linspace(1, min(width, height)//2, 20)
        K_range = [calculate_stress_intensity_factor(crack_type, applied_stress, a, width, height) 
                   for a in crack_range]
        
        fig_param = go.Figure()
        fig_param.add_trace(go.Scatter(x=crack_range, y=K_range, mode='lines+markers',
                                      name='K_I', line=dict(color='cyan')))
        fig_param.add_hline(y=K_IC, line_dash="dash", line_color="red",
                           annotation_text=f"K_IC = {K_IC} MPa√m")
        fig_param.update_layout(
            title="Stress Intensity Factor vs Crack Length",
            xaxis_title="Crack Length (mm)",
            yaxis_title="K_I (MPa√m)",
            plot_bgcolor='#2E2E2E',
            paper_bgcolor='#2E2E2E',
            font=dict(color='white')
        )
        st.plotly_chart(fig_param, use_container_width=True)

with tab2:
    st.subheader("Fatigue Crack Growth Analysis")
    
    # Paris law parameters
    st.write("**Paris Law:** da/dN = C(ΔK)^m")
    
    # Create fatigue analysis
    cycles_range = np.logspace(1, 6, 100)
    crack_lengths_fatigue = []
    current_crack = crack_length
    
    for cycle in cycles_range:
        K_current = calculate_stress_intensity_factor(crack_type, applied_stress, current_crack, width, height)
        growth, _ = paris_law_crack_growth(K_current, delta_K_threshold, 1)
        current_crack += growth
        crack_lengths_fatigue.append(current_crack)
        
        if current_crack > min(width, height)//2:
            break
    
    fig_fatigue = go.Figure()
    fig_fatigue.add_trace(go.Scatter(x=cycles_range[:len(crack_lengths_fatigue)], 
                                    y=crack_lengths_fatigue, mode='lines',
                                    name='Crack Growth', line=dict(color='yellow')))
    fig_fatigue.update_layout(
        title="Fatigue Crack Growth",
        xaxis_title="Cycles",
        yaxis_title="Crack Length (mm)",
        xaxis_type="log",
        plot_bgcolor='#2E2E2E',
        paper_bgcolor='#2E2E2E',
        font=dict(color='white')
    )
    st.plotly_chart(fig_fatigue, use_container_width=True)

with tab3:
    st.subheader("Failure Assessment Diagram")
    
    # Create failure assessment diagram
    Kr = K_I / K_IC  # Ratio of applied to critical stress intensity
    Sr = applied_stress / yield_strength  # Ratio of applied to yield stress
    
    # Assessment curve (simplified)
    Sr_range = np.linspace(0, 1, 100)
    Kr_limit = np.sqrt(1 - Sr_range)  # Simplified assessment curve
    
    fig_fad = go.Figure()
    fig_fad.add_trace(go.Scatter(x=Sr_range, y=Kr_limit, mode='lines',
                                name='Assessment Curve', line=dict(color='red')))
    fig_fad.add_trace(go.Scatter(x=[Sr], y=[Kr], mode='markers',
                                name='Operating Point', marker=dict(color='yellow', size=10)))
    
    fig_fad.update_layout(
        title="Failure Assessment Diagram",
        xaxis_title="Sr (σ/σy)",
        yaxis_title="Kr (K/K_IC)",
        plot_bgcolor='#2E2E2E',
        paper_bgcolor='#2E2E2E',
        font=dict(color='white')
    )
    st.plotly_chart(fig_fad, use_container_width=True)
    
    if Kr < np.sqrt(1 - Sr):
        st.success("✅ Operating point is within safe region")
    else:
        st.error("⚠️ Operating point exceeds safe region - Failure expected!")

st.markdown("---")
st.markdown("*This application provides advanced fracture mechanics analysis for crack propagation studies.*")
