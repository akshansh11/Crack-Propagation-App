import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle, Polygon
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import time

# Set page config
st.set_page_config(page_title="Crack Propagation Analysis", layout="wide")

# Custom CSS for clean white theme
st.markdown("""
<style>
    .main {
        background-color: #FFFFFF;
        color: #000000;
    }
    .stApp {
        background-color: #FFFFFF;
    }
    .sidebar .sidebar-content {
        background-color: #F8F9FA;
    }
    h1, h2, h3 {
        color: #2C3E50;
    }
    .stSelectbox > div > div {
        background-color: #FFFFFF;
        color: #000000;
    }
    .stSlider > div > div > div {
        background-color: #FFFFFF;
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
    
    # Initialize default values
    E = 200.0
    nu = 0.3
    yield_strength = 800.0
    K_IC = 50.0
    
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
    applied_stress = st.slider("Applied Stress (MPa)", 10, int(yield_strength), min(200, int(yield_strength//2)))
    
    if stress_type == "Mixed Mode":
        mode_I_ratio = st.slider("Mode I Ratio", 0.0, 1.0, 0.7)
        mode_II_ratio = 1 - mode_I_ratio
    
    st.header("Analysis Parameters")
    num_cycles = st.slider("Number of Load Cycles", 1, 10000, 1000)
    delta_K_threshold = st.slider("ΔK Threshold (MPa√m)", 1.0, 10.0, 3.0)
    
    st.header("Animation Controls")
    animate_crack = st.checkbox("Enable Crack Propagation Animation", value=True)
    animation_speed = st.slider("Animation Speed", 0.1, 2.0, 1.0, 0.1)
    animation_steps = st.slider("Animation Steps", 5, 50, 20)

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

    # Calculate crack growth history for animation
    def calculate_crack_growth_history(initial_crack, num_steps, total_cycles):
        """Calculate crack growth history for animation"""
        history = []
        current_crack = initial_crack
        cycles_per_step = total_cycles // num_steps
        
        for step in range(num_steps + 1):
            current_cycles = step * cycles_per_step
            K_current = calculate_stress_intensity_factor(crack_type, applied_stress, current_crack, width, height)
            
            if K_current < K_IC:  # Only grow if below critical
                growth, da_dN = paris_law_crack_growth(K_current, delta_K_threshold, cycles_per_step)
                current_crack += growth
                
                # Limit crack growth to reasonable bounds
                max_crack = min(width, height) // 3
                current_crack = min(current_crack, max_crack)
            
            history.append({
                'cycle': current_cycles,
                'crack_length': current_crack,
                'K_I': K_current,
                'step': step
            })
            
        return history
    
    # Calculate current values
    K_I = calculate_stress_intensity_factor(crack_type, applied_stress, crack_length, width, height)
    crack_growth, da_dN = paris_law_crack_growth(K_I, delta_K_threshold, num_cycles)
    
    # Calculate crack growth history for animation
    if animate_crack:
        crack_history = calculate_crack_growth_history(crack_length, animation_steps, num_cycles)
    else:
        crack_history = [{'cycle': 0, 'crack_length': crack_length, 'K_I': K_I, 'step': 0}]
    
    # Safety factor
    safety_factor = K_IC / K_I if K_I > 0 else float('inf')
    
    # Create animated visualization
    if animate_crack and len(crack_history) > 1:
        # Create animation frames
        frames = []
        
        for i, point in enumerate(crack_history):
            current_crack_length = point['crack_length']
            current_K_I = point['K_I']
            current_cycle = point['cycle']
            
            # Crack geometry for current frame
            if crack_type == "Center Crack":
                x_crack = np.linspace(-current_crack_length/2, current_crack_length/2, 100)
            elif crack_type == "Edge Crack":
                x_crack = np.linspace(-width/2, -width/2 + current_crack_length, 100)
            else:  # Corner or penny-shaped
                x_crack = np.linspace(-width/2, -width/2 + current_crack_length, 100)
            
            y_crack = np.zeros_like(x_crack)
            
            # Create frame data
            frame_data = []
            
            # Specimen outline
            frame_data.append(go.Scatter(
                x=[-width/2, width/2, width/2, -width/2, -width/2],
                y=[-height/2, -height/2, height/2, height/2, -height/2],
                mode='lines', name='Specimen', 
                line=dict(color='black', width=2),
                showlegend=(i==0)
            ))
            
            # Crack
            frame_data.append(go.Scatter(
                x=x_crack, y=y_crack, mode='lines', name='Crack',
                line=dict(color='red', width=6),
                showlegend=(i==0)
            ))
            
            # Stress field visualization (optional)
            if crack_type == "Center Crack":
                # Add stress concentration visualization
                theta = np.linspace(0, 2*np.pi, 50)
                stress_radius = current_crack_length * 2
                stress_x = stress_radius * np.cos(theta)
                stress_y = stress_radius * np.sin(theta)
                
                frame_data.append(go.Scatter(
                    x=stress_x, y=stress_y, mode='lines',
                    name='Stress Field', line=dict(color='blue', width=1, dash='dot'),
                    opacity=0.5, showlegend=(i==0)
                ))
            
            frames.append(go.Frame(
                data=frame_data,
                name=str(i),
                layout=go.Layout(
                    title=f"Crack Propagation - Cycle: {current_cycle:,} | "
                          f"Crack Length: {current_crack_length:.2f} mm | "
                          f"K_I: {current_K_I:.2f} MPa√m"
                )
            ))
        
        # Create initial figure
        initial_crack_length = crack_history[0]['crack_length']
        if crack_type == "Center Crack":
            x_crack_init = np.linspace(-initial_crack_length/2, initial_crack_length/2, 100)
        elif crack_type == "Edge Crack":
            x_crack_init = np.linspace(-width/2, -width/2 + initial_crack_length, 100)
        else:
            x_crack_init = np.linspace(-width/2, -width/2 + initial_crack_length, 100)
        y_crack_init = np.zeros_like(x_crack_init)
        
        fig_anim = go.Figure(
            data=[
                go.Scatter(
                    x=[-width/2, width/2, width/2, -width/2, -width/2],
                    y=[-height/2, -height/2, height/2, height/2, -height/2],
                    mode='lines', name='Specimen', line=dict(color='black', width=2)
                ),
                go.Scatter(
                    x=x_crack_init, y=y_crack_init, mode='lines', name='Crack',
                    line=dict(color='red', width=6)
                )
            ],
            frames=frames
        )
        
        # Animation controls
        fig_anim.update_layout(
            title=f"Animated Crack Propagation - {crack_type}",
            xaxis_title="Position (mm)",
            yaxis_title="Position (mm)",
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF',
            font=dict(color='black'),
            height=600,
            xaxis=dict(range=[-width/2*1.2, width/2*1.2], gridcolor='#E0E0E0'),
            yaxis=dict(range=[-height/2*1.2, height/2*1.2], gridcolor='#E0E0E0'),
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": int(1000/animation_speed), "redraw": True},
                                       "fromcurrent": True, "transition": {"duration": 300}}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                         "mode": "immediate", "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Step:",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [[f.name], {"frame": {"duration": 300, "redraw": True},
                                           "mode": "immediate", "transition": {"duration": 300}}],
                        "label": str(i),
                        "method": "animate"
                    }
                    for i, f in enumerate(frames)
                ]
            }]
        )
        
        st.plotly_chart(fig_anim, use_container_width=True)
        
        # Display animation info
        st.info(f"🎬 Animation shows crack growth over {num_cycles:,} cycles in {animation_steps} steps")
        
    else:
        # Static visualization (original)
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
                       mode='lines', name='Specimen', line=dict(color='black', width=2)),
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
                       line=dict(color='blue', width=2)),
            row=1, col=2
        )
        
        fig.add_hline(y=K_IC, line_dash="dash", line_color="red", 
                      annotation_text=f"K_IC = {K_IC} MPa√m", row=1, col=2)
        
        # Crack growth rate plot
        cycles = np.linspace(1, num_cycles, 100)
        growth_rates = [paris_law_crack_growth(K_I, delta_K_threshold, n)[0] for n in cycles]
        
        fig.add_trace(
            go.Scatter(x=cycles, y=growth_rates, mode='lines', name='Crack Growth',
                       line=dict(color='green', width=2)),
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
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF',
            font=dict(color='black'),
            height=800
        )
        
        # Update axes
        for i in range(1, 3):
            for j in range(1, 3):
                fig.update_xaxes(gridcolor='#E0E0E0', row=i, col=j)
                fig.update_yaxes(gridcolor='#E0E0E0', row=i, col=j)
        
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
                                      name='K_I', line=dict(color='blue')))
        fig_param.add_hline(y=K_IC, line_dash="dash", line_color="red",
                           annotation_text=f"K_IC = {K_IC} MPa√m")
        fig_param.update_layout(
            title="Stress Intensity Factor vs Crack Length",
            xaxis_title="Crack Length (mm)",
            yaxis_title="K_I (MPa√m)",
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF',
            font=dict(color='black')
        )
        st.plotly_chart(fig_param, use_container_width=True)

with tab2:
    st.subheader("Fatigue Crack Growth Analysis")
    
    # Show animation data if available
    if animate_crack and len(crack_history) > 1:
        # Create DataFrame from crack history
        df_history = pd.DataFrame(crack_history)
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            # Plot crack growth over cycles
            fig_growth = go.Figure()
            fig_growth.add_trace(go.Scatter(
                x=df_history['cycle'], 
                y=df_history['crack_length'],
                mode='lines+markers',
                name='Crack Length',
                line=dict(color='red', width=3),
                marker=dict(size=6)
            ))
            fig_growth.update_layout(
                title="Crack Length vs Cycles",
                xaxis_title="Cycles",
                yaxis_title="Crack Length (mm)",
                plot_bgcolor='#FFFFFF',
                paper_bgcolor='#FFFFFF',
                font=dict(color='black'),
                height=400
            )
            st.plotly_chart(fig_growth, use_container_width=True)
        
        with col_b:
            # Plot K_I evolution
            fig_ki = go.Figure()
            fig_ki.add_trace(go.Scatter(
                x=df_history['cycle'], 
                y=df_history['K_I'],
                mode='lines+markers',
                name='K_I',
                line=dict(color='blue', width=3),
                marker=dict(size=6)
            ))
            fig_ki.add_hline(y=K_IC, line_dash="dash", line_color="red",
                           annotation_text=f"K_IC = {K_IC} MPa√m")
            fig_ki.update_layout(
                title="Stress Intensity Factor vs Cycles",
                xaxis_title="Cycles",
                yaxis_title="K_I (MPa√m)",
                plot_bgcolor='#FFFFFF',
                paper_bgcolor='#FFFFFF',
                font=dict(color='black'),
                height=400
            )
            st.plotly_chart(fig_ki, use_container_width=True)
        
        # Display crack growth table
        st.subheader("Crack Growth History")
        df_display = df_history.copy()
        df_display['cycle'] = df_display['cycle'].astype(int)
        df_display['crack_length'] = df_display['crack_length'].round(3)
        df_display['K_I'] = df_display['K_I'].round(2)
        df_display = df_display.drop('step', axis=1)
        st.dataframe(df_display, use_container_width=True)
    
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
                                    name='Crack Growth', line=dict(color='green')))
    fig_fatigue.update_layout(
        title="Fatigue Crack Growth (Paris Law)",
        xaxis_title="Cycles",
        yaxis_title="Crack Length (mm)",
        xaxis_type="log",
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        font=dict(color='black')
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
                                name='Operating Point', marker=dict(color='orange', size=10)))
    
    fig_fad.update_layout(
        title="Failure Assessment Diagram",
        xaxis_title="Sr (σ/σy)",
        yaxis_title="Kr (K/K_IC)",
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        font=dict(color='black')
    )
    st.plotly_chart(fig_fad, use_container_width=True)
    
    if Kr < np.sqrt(1 - Sr):
        st.success("✅ Operating point is within safe region")
    else:
        st.error("⚠️ Operating point exceeds safe region - Failure expected!")

st.markdown("---")
st.markdown("*This application provides advanced fracture mechanics analysis for crack propagation studies.*")
