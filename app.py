import os
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
from dotenv import load_dotenv

load_dotenv()

# Sample Data Loading Function
def load_sample_data():
    sample_data = """Date	Product_ID	Category	Sales_Quantity	Price	Promotion_Flag	Holiday_Flag	Market_Trend_Score	Current_Stock	Reorder_Threshold	Lead_Time_Days	Supplier_ID	Supplier_Reliability_Score	Demand_Forecast_Next_30_Days	Risk_Type	Severity_Level	Reason	Expected_Delay_Days
2023-01-01	1078	Jeans	74	44.4	1	1	92	842	141	10	238	67	1712	No Risk	Low	Port Strikes	5
2023-01-02	1041	Jackets	366	52.64	0	1	85	947	100	12	201	86	908	Shipment Delay	High	Port Strikes	8
2023-01-03	1063	Jeans	440	28.46	1	0	51	354	274	12	240	68	1025	No Risk	Medium	Port Strikes	6
2023-01-04	1060	Jackets	133	95.96	0	0	74	803	241	9	234	71	1889	No Risk	Low	Port Strikes	1
2023-01-05	1016	Shirts	113	20.8	0	1	74	342	77	8	229	93	1424	No Risk	High	No Issue	7
2023-01-06	1019	Jeans	346	99.89	0	1	96	163	101	2	248	81	684	No Risk	Low	High Demand	4
2023-01-07	1072	T-Shirts	251	149.16	1	0	50	396	278	2	297	70	697	No Risk	Low	Port Strikes	7
2023-01-08	1057	Shirts	351	59.62	0	1	84	620	63	8	247	85	801	Material Shortage	High	High Demand	1
2023-01-09	1052	Dresses	329	73.08	1	0	94	200	62	9	225	62	1566	Material Shortage	Low	Factory Shutdown	2
2023-01-10	1023	Shirts	111	13.73	0	1	94	742	84	8	230	85	780	No Risk	High	No Issue	0
2023-01-11	1055	Shirts	90	10.83	1	0	54	983	240	6	256	88	1697	No Risk	Medium	High Demand	1
2023-01-12	1091	Jackets	434	106.74	1	0	60	216	110	12	271	76	1557	No Risk	Low	No Issue	5
2023-01-13	1048	Dresses	384	144.73	0	0	86	620	168	10	259	74	1153	No Risk	High	High Demand	5
2023-01-14	1087	T-Shirts	319	55.48	1	1	95	756	123	7	229	98	638	No Risk	Medium	Port Strikes	0
2023-01-15	1066	Jeans	149	40.22	0	1	88	932	247	13	250	77	1051	No Risk	High	Factory Shutdown	2
2023-01-16	1057	Jackets	206	34.73	0	1	89	132	78	6	276	91	897	Shipment Delay	High	Port Strikes	7
2023-01-17	1059	T-Shirts	489	96.44	1	0	93	910	80	3	260	85	1829	Material Shortage	High	No Issue	9
2023-01-18	1008	Shirts	150	26.82	0	1	94	488	158	11	262	74	348	No Risk	High	No Issue	6
2023-01-19	1089	T-Shirts	209	149.52	1	0	88	152	68	9	274	91	1909	Material Shortage	Medium	No Issue	8
2023-01-20	1039	Jeans	53	57.28	0	0	62	136	109	7	267	92	1993	No Risk	Low	Factory Shutdown	5
2023-01-21	1034	Jeans	392	41.62	0	1	87	264	138	13	220	78	1586	No Risk	Medium	Port Strikes	0
2023-01-22	1014	Jackets	414	148.79	0	0	94	425	72	13	295	92	1753	No Risk	Medium	No Issue	4
2023-01-23	1074	Dresses	229	31.96	0	1	69	747	263	5	248	69	1800	No Risk	Low	No Issue	4
2023-01-24	1041	Jackets	488	109.25	1	0	63	499	269	10	227	86	1345	No Risk	Medium	Factory Shutdown	2
2023-01-25	1087	Jeans	390	14.39	1	1	80	169	131	2	261	83	1084	No Risk	Medium	High Demand	6
2023-01-26	1020	Dresses	473	58.88	1	1	65	206	138	2	241	95	1635	No Risk	High	No Issue	3
2023-01-27	1024	Dresses	485	21.09	1	0	96	756	180	5	250	77	813	No Risk	Medium	Port Strikes	3
2023-01-28	1035	Shirts	218	31.5	0	1	73	939	217	4	278	79	729	No Risk	High	High Demand	8
2023-01-29	1045	Jackets	311	102.29	0	0	74	966	87	9	258	98	495	No Risk	Medium	No Issue	8
2023-01-30	1028	T-Shirts	446	148.99	1	1	76	838	85	11	269	82	641	No Risk	High	No Issue	1"""
    return pd.read_csv(StringIO(sample_data), sep='\t', parse_dates=['Date'])

# IBM Granite Helper Functions
def get_ibm_token():
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = {
        'grant_type': 'urn:ibm:params:oauth:grant-type:apikey',
        'apikey': os.getenv("IBM_API_KEY")
    }
    response = requests.post(url, headers=headers, data=data)
    return response.json()['access_token']

def granite_query(prompt):
    try:
        token = get_ibm_token()
        url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {token}'
        }
        payload = {
            "input": f"<|start_of_role|>system<|end_of_role|>You are a senior supply chain analyst. Provide detailed explanations with numerical evidence and actionable recommendations.<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>{prompt}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>",
            "parameters": {"max_new_tokens": 5000},
            "model_id": "ibm/granite-3-8b-instruct",
            "project_id": os.getenv("PROJECT_ID")
        }
        response = requests.post(url, headers=headers, json=payload)
        return response.json()['results'][0]['generated_text']
    except Exception as e:
        return f"Error generating insights: {str(e)}"

# Data Processing
def process_data(df):
    df['Sales_Revenue'] = df['Sales_Quantity'] * df['Price']
    category_df = df.groupby('Category').agg({
        'Sales_Quantity': 'sum',
        'Sales_Revenue': 'sum',
        'Current_Stock': 'sum',
        'Demand_Forecast_Next_30_Days': 'sum',
        'Lead_Time_Days': 'mean',
        'Supplier_Reliability_Score': 'mean',
        'Expected_Delay_Days': 'mean'
    }).reset_index()
    return df, category_df

# Main App
def main():
    st.set_page_config(page_title="SyncChain - AI Supply Chain Analytics", layout="wide")
    
    # Initialize session state
    if 'show_main_app' not in st.session_state:
        st.session_state.show_main_app = False
    
    # Show landing page or main app based on session state
    if not st.session_state.show_main_app:
        landing_page()
    else:
        # Header
        st.title("🚚 SyncChain - AI Powered Supply Chain Analyst")
        st.markdown("IBM Granite-powered supply chain optimization with XAI explanations")
        
        # Add a button to return to landing page
        if st.sidebar.button("← Back to Home"):
            st.session_state.show_main_app = False
            st.rerun()
        
        # Data Loading
        st.sidebar.header("Data Input")
        data_source = st.sidebar.radio("Choose Data Source:", 
                                     ["Upload CSV", "Use Sample Data"])
        
        if data_source == "Upload CSV":
            uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
            if uploaded_file:
                df = pd.read_csv(uploaded_file, parse_dates=['Date'])
            else:
                st.info("📁 Please upload a CSV file or select 'Use Sample Data'")
                st.stop()
        else:
            df = load_sample_data()
            st.sidebar.success("✅ Using sample dataset with realistic supply chain scenarios")
        
        # Process data
        df, category_df = process_data(df)
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Demand Forecast", 
            "📦 Inventory Optimizer", 
            "🚨 Risk Monitor", 
            "🎮 Scenario Lab"
        ])

        # Tab 1: Demand Forecast
        with tab1:
            st.header("Demand Forecasting Engine")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                selected_category = st.selectbox("Select Category", 
                                                category_df['Category'].unique(),
                                                key='cat_select')
                category_data = category_df[category_df['Category'] == selected_category]
                filtered_df = df[df['Category'] == selected_category]
                
                st.metric("Total Monthly Revenue", 
                         f"${category_data['Sales_Revenue'].values[0]:,.2f}")
                st.metric("30-Day Demand Forecast", 
                         f"{category_data['Demand_Forecast_Next_30_Days'].values[0]:,.0f} units")

            with col2:
                with st.spinner("Analyzing demand drivers..."):
                    prompt = f"""
                    Analyze demand for {selected_category} category with:
                    - 30-day sales trend: {filtered_df['Sales_Quantity'].tail(30).tolist()}
                    - Average price: ${filtered_df['Price'].mean():.2f}
                    - Promotion days: {filtered_df['Promotion_Flag'].sum()}
                    - Market trend score: {filtered_df['Market_Trend_Score'].mean()}/100
                    - Holiday impact: {filtered_df[filtered_df['Holiday_Flag'] == 1]['Sales_Quantity'].mean()/filtered_df['Sales_Quantity'].mean():.1%}
                    Provide XAI explanation with percentage contributions and forecast confidence.
                    """
                    analysis = granite_query(prompt)
                    st.write(analysis)
            
            fig = px.line(filtered_df, x='Date', y='Sales_Quantity',
                         title=f"{selected_category} Demand Pattern",
                         template='plotly_white',
                         markers=True)
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        # Tab 2: Inventory Optimizer
        with tab2:
            st.header("Inventory Optimization Dashboard")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Key Metrics")
                st.metric("Current Stock", f"{category_data['Current_Stock'].values[0]:,.0f}")
                st.metric("Avg Lead Time", f"{category_data['Lead_Time_Days'].values[0]:.1f} days")
                st.metric("Supplier Reliability", f"{category_data['Supplier_Reliability_Score'].values[0]:.0f}/100")

            with col2:
                with st.spinner("Generating recommendations..."):
                    prompt = f"""
                    Recommend inventory for {selected_category} with:
                    - Current stock: {category_data['Current_Stock'].values[0]:,.0f}
                    - Lead time: {category_data['Lead_Time_Days'].values[0]} days
                    - Supplier score: {category_data['Supplier_Reliability_Score'].values[0]}/100
                    - Demand forecast: {category_data['Demand_Forecast_Next_30_Days'].values[0]:,.0f}
                    Include safety stock calculation and reorder points with numerical values.
                    """
                    analysis = granite_query(prompt)
                    st.write(analysis)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Current', 'Recommended'],
                y=[category_data['Current_Stock'].values[0], 
                category_data['Current_Stock'].values[0] * 1.3],
                marker_color=['#636EFA', '#00CC96']
            ))
            fig.update_layout(title='Inventory Position Analysis',
                             template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

        # Tab 3: Risk Monitor
        with tab3:
            st.header("Risk Intelligence Center")
            risk_df = df.groupby(['Risk_Type', 'Severity_Level']).size().reset_index(name='Count')
            
            col1, col2 = st.columns([1, 2])
            with col1:
                fig = px.treemap(risk_df, path=['Severity_Level', 'Risk_Type'], values='Count',
                                 color='Severity_Level', color_discrete_map={
                                     'High':'red', 'Medium':'orange', 'Low':'green'},
                                 title='Risk Distribution')
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                with st.spinner("Analyzing risks..."):
                    prompt = f"""
                    Analyze risks from: {risk_df.to_dict()}
                    Identify top 3 critical risks with root causes and mitigation timelines.
                    Include probability estimates and financial impact projections.
                    """
                    analysis = granite_query(prompt)
                    st.write(analysis)

        # Tab 4: Scenario Lab
        with tab4:
            st.header("Scenario Simulation Lab")
            with st.form("scenario_form"):
                col1, col2 = st.columns(2)
                with col1:
                    demand_change = st.slider("Demand Change (%)", -50, 100, 0)
                    lead_time_change = st.slider("Lead Time Change (days)", -5, 15, 0)
                with col2:
                    promo_days = st.slider("Promotion Days", 0, 30, 0)
                    risk_threshold = st.slider("Risk Threshold (%)", 0, 100, 70)
                
                if st.form_submit_button("Run Simulation"):
                    with st.spinner("Simulating scenario..."):
                        prompt = f"""
                        Simulate scenario:
                        - Demand change: {demand_change}%
                        - Lead time change: {lead_time_change} days
                        - Promotion days: {promo_days}
                        - Risk threshold: {risk_threshold}%
                        Base parameters: {category_data.iloc[0].to_dict()}
                        Predict impacts on:
                        1. Stockout probability
                        2. Inventory costs
                        3. Delivery reliability
                        4. Required safety stock
                        Provide numerical projections with confidence intervals.
                        """
                        analysis = granite_query(prompt)
                        st.write(analysis)

def landing_page():


    # Add spacing to prevent content overlap
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
     
    # Advanced CSS with modern design elements and color schemes
    st.markdown("""
    <style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Inter:wght@400;500;600&display=swap');
    
    /* Reset and base styles */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        background: #1a1a1a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #84fab0, #8fd3f4);
        border-radius: 5px;
    }
    
    /* Main background with animated gradient */
    .stApp {
        background: linear-gradient(
            -45deg,
            #0f172a,
            #1e293b,
            #242f3f,
            #334155
        );
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    
    @keyframes gradient {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }
    
    /* Header styles with 3D effect */
    .main-header {
        text-align: center;
        padding: 4rem 0;
        background: rgba(0, 0, 0, 0.2);
        border-radius: 20px;
        margin: 20px;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(255, 255, 255, 0.2),
            transparent
        );
        transition: 0.5s;
    }
    
    .main-header:hover::before {
        left: 100%;
    }
    
    .main-header h1 {
        font-family: 'Poppins', sans-serif;
        font-size: 4.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        transform: perspective(500px) rotateX(0deg);
        transition: transform 0.3s ease;
    }
    
    .main-header h1:hover {
        transform: perspective(500px) rotateX(5deg);
    }
    
    /* Animated subtitle */
    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        color: #94a3b8;
        margin-bottom: 2rem;
        font-weight: 400;
        line-height: 1.6;
        opacity: 0;
        animation: fadeInUp 1s ease forwards;
        animation-delay: 0.5s;
    }
    
    /* Enhanced feature cards */
    .feature-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 30px;
        padding: 40px 20px;
        margin: 2rem auto;
        max-width: 1400px;
        perspective: 1000px;
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 35px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
        transform-style: preserve-3d;
        transform: translateZ(0);
    }
    
    .feature-card::before,
    .feature-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        border-radius: 20px;
        background: linear-gradient(45deg, 
            rgba(132, 250, 176, 0.1),
            rgba(143, 211, 244, 0.1)
        );
        z-index: -1;
        transition: opacity 0.3s ease;
    }
    
    .feature-card::after {
        background: linear-gradient(45deg,
            rgba(132, 250, 176, 0.2),
            rgba(143, 211, 244, 0.2)
        );
        opacity: 0;
    }
    
    .feature-card:hover {
        transform: translateY(-10px) rotateX(5deg);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
    }
    
    .feature-card:hover::after {
        opacity: 1;
    }
    
    /* Floating animation for stats */
    .stat-item {
        animation: float 6s ease-in-out infinite;
    }
    
    .stat-item:nth-child(2) {
        animation-delay: 1s;
    }
    
    .stat-item:nth-child(3) {
        animation-delay: 2s;
    }
    
    .stat-item:nth-child(4) {
        animation-delay: 3s;
    }
    
    /* Enhanced CTA button */
    .cta-button {
        background: linear-gradient(
            45deg,
            #84fab0 0%,
            #8fd3f4 100%
        );
        color: #1a1a1a;
        padding: 18px 40px;
        border-radius: 50px;
        font-size: 1.3rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.4s ease;
        border: none;
        box-shadow: 0 10px 20px rgba(132, 250, 176, 0.2);
        position: relative;
        overflow: hidden;
        z-index: 1;
    }
    
    .cta-button::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            45deg,
            #8fd3f4 0%,
            #84fab0 100%
        );
        opacity: 0;
        z-index: -1;
        transition: opacity 0.4s ease;
    }
    
    .cta-button:hover::before {
        opacity: 1;
    }
    
    .cta-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 30px rgba(132, 250, 176, 0.3);
    }
    
    /* New feature: Animated background particles */
    .particles {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
        overflow: hidden;
    }
    
    .particle {
        position: absolute;
        width: 2px;
        height: 2px;
        background: rgba(255, 255, 255, 0.5);
        border-radius: 50%;
        animation: particleFloat 20s infinite linear;
    }
    
    @keyframes particleFloat {
        0% {
            transform: translateY(0) translateX(0);
            opacity: 0;
        }
        50% {
            opacity: 1;
        }
        100% {
            transform: translateY(-100vh) translateX(100vw);
            opacity: 0;
        }
    }
    
    /* New feature: Glowing text effect */
    .glow-text {
        text-shadow: 0 0 10px rgba(132, 250, 176, 0.5),
                     0 0 20px rgba(132, 250, 176, 0.3),
                     0 0 30px rgba(132, 250, 176, 0.1);
    }
    
    /* New feature: Pulse animation */
    @keyframes pulse {
        0% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
        100% {
            transform: scale(1);
        }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* Enhanced responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2.8rem;
        }
        .subtitle {
            font-size: 1.2rem;
            padding: 0 20px;
        }
        .feature-card {
            margin: 10px;
        }
        .stat-number {
            font-size: 2.5rem;
        }
    }
    
    /* New animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes float {
        0%, 100% {
            transform: translateY(0);
        }
        50% {
            transform: translateY(-10px);
        }
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 5rem;
        font-weight: 700;
        background: linear-gradient(120deg, 
            #00ffcc 0%, 
            #00e6ff 50%, 
            #33ccff 100%
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem;
        position: relative;
        animation: titleFloat 6s ease-in-out infinite;
        text-shadow: 
            0 0 20px rgba(0, 255, 204, 0.5),
            0 0 40px rgba(0, 230, 255, 0.3),
            0 0 60px rgba(51, 204, 255, 0.2);
    }

    /* Add this glow animation */
    @keyframes glow {
        0%, 100% {
            text-shadow: 
                0 0 20px rgba(0, 255, 204, 0.5),
                0 0 40px rgba(0, 230, 255, 0.3),
                0 0 60px rgba(51, 204, 255, 0.2);
        }
        50% {
            text-shadow: 
                0 0 30px rgba(0, 255, 204, 0.7),
                0 0 60px rgba(0, 230, 255, 0.5),
                0 0 90px rgba(51, 204, 255, 0.3);
        }
    }

    /* Add this to the hero-section class */
    .hero-section {
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 30px;
        padding: 4rem 2rem;
        margin: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 
            0 20px 40px rgba(0,0,0,0.2),
            0 0 30px rgba(0, 255, 204, 0.1),
            0 0 60px rgba(0, 230, 255, 0.1);
    }
    </style>
    
    <!-- Particle effect -->
    <div class="particles">
        ${Array(20).fill().map((_, i) => `
            <div class="particle" style="
                left: ${Math.random() * 100}vw;
                animation-delay: ${Math.random() * 20}s;
                animation-duration: ${15 + Math.random() * 10}s;
            "></div>
        `).join('')}
    </div>
    """, unsafe_allow_html=True)

    # Rest of your landing page content remains the same
    # Enhanced Header Section with glow effect
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">🔄 ChainSync Analytics</h1>
        <p style="text-align: center; color: #94a3b8; font-size: 1.5rem; margin-bottom: 2rem;">
            Revolutionize Your Supply Chain with Next-Generation AI Intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)



    # After your title section and before feature cards, add the embedded video
    st.markdown("""
    <style>
    /* Video container styling */
    .video-container {
        position: relative;
        width: 100%;
        max-width: 800px;
        margin: 3rem auto;
        padding: 0 20px;
    }

    .video-wrapper {
        position: relative;
        padding-bottom: 56.25%; /* 16:9 Aspect Ratio */
        height: 0;
        overflow: hidden;
        border-radius: 16px;
        box-shadow: 
            0 8px 30px rgba(0, 0, 0, 0.3),
            0 0 40px rgba(45, 211, 255, 0.2);
    }

    .video-wrapper iframe {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        border: none;
        border-radius: 16px;
    }

    .video-title {
        text-align: center;
        font-size: 1.8rem;
        color: #E0F7FF;
        margin: 2rem 0;
        font-weight: 600;
        text-shadow: 0 2px 10px rgba(0, 255, 255, 0.2);
    }
    </style>

    <div class="video-container">
        <h2 class="video-title">Watch Our Demo</h2>
        <div class="video-wrapper">
            <iframe
                src="https://www.youtube.com/embed/9ZYpQvWEnjk"
                title="ChainSync Analytics Demo"
                frameborder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowfullscreen>
            </iframe>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Enhanced Feature Cards
    st.markdown("""
    <div class="feature-container">
        <div class="feature-card">
            <h3>📈 Predictive Analytics</h3>
            <p>Harness the power of AI to forecast demand patterns with unprecedented accuracy. Our advanced algorithms provide 95% accurate predictions.</p>
        </div>
        <div class="feature-card">
            <h3>🎯 Smart Inventory</h3>
            <p>Optimize stock levels automatically with real-time tracking and AI-driven reordering systems. Reduce holding costs by up to 30%.</p>
        </div>
        <div class="feature-card">
            <h3>🛡️ Risk Intelligence</h3>
            <p>Proactively identify and mitigate supply chain risks with our advanced early warning system and automated contingency planning.</p>
        </div>
        <div class="feature-card">
            <h3>🔄 Scenario Planning</h3>
            <p>Create and simulate multiple supply chain scenarios in real-time. Make data-driven decisions with confidence.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Enhanced Stats Section
    st.markdown("""
    <div class="stats-container">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;">
            <div class="stat-item">
                <div class="stat-number">95%</div>
                <div class="stat-label">Forecast Accuracy</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">30%</div>
                <div class="stat-label">Cost Reduction</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">24/7</div>
                <div class="stat-label">Real-time Monitoring</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">50+</div>
                <div class="stat-label">Enterprise Clients</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Enhanced Benefits Section
    st.markdown("""
    <div class="benefits-container">
        <h2 style="text-align: center; color: #f1f5f9; margin-bottom: 40px; font-size: 2.5rem;">
            Why Choose SyncChain?
        </h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 30px;">
            <div class="benefit-item">
                <h3 style="color: #f1f5f9; margin-bottom: 15px;">🤖 AI-Powered</h3>
                <p style="color: #94a3b8;">Cutting-edge IBM Granite integration delivering intelligent insights and predictions</p>
            </div>
            <div class="benefit-item">
                <h3 style="color: #f1f5f9; margin-bottom: 15px;">📊 Real-Time Analytics</h3>
                <p style="color: #94a3b8;">Live dashboards and instant alerts for immediate decision-making</p>
            </div>
            <div class="benefit-item">
                <h3 style="color: #f1f5f9; margin-bottom: 15px;">🔐 Enterprise Security</h3>
                <p style="color: #94a3b8;">Bank-grade encryption and security protocols protecting your data</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

   # Enhanced CTA Section with reduced spacing
    st.markdown("""
    <div style="text-align: center; padding: 30px 20px;">
        <h2 style="color: #f1f5f9; margin-bottom: 15px; font-size: 2.5rem;">
            Ready to Transform Your Supply Chain?
        </h2>
        <h4 style="color: #94a3b8; font-size: 1.2rem; margin-bottom: 20px;">
            Join industry leaders who have revolutionized their operations with SyncChain
        </h4>
    </div>

    <style>
    /* Reduce spacing between sections */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Adjust vertical gaps */
    .stMarkdown {
        margin-bottom: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    /* File viewer container styling */
    .file-container {
        position: relative;
        width: 100%;
        max-width: 800px;
        margin: 3rem auto;
        padding: 0 20px;
    }

    .file-wrapper {
        position: relative;
        padding-bottom: 75%; /* 4:3 Aspect Ratio */
        height: 0;
        overflow: hidden;
        border-radius: 16px;
        box-shadow: 
            0 8px 30px rgba(0, 0, 0, 0.3),
            0 0 40px rgba(45, 211, 255, 0.2);
    }

    .file-wrapper iframe {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        border: none;
        border-radius: 16px;
    }

    .file-title {
        text-align: center;
        font-size: 1.8rem;
        color: #E0F7FF;
        margin: 2rem 0;
        font-weight: 600;
        text-shadow: 0 2px 10px rgba(0, 255, 255, 0.2);
    }
    </style>

    <div class="file-container">
        <h2 class="file-title">Sample CSV File</h2>
        <h4 class="file-title">Click on the button below to download the sample CSV file.</h4>
        <div class="file-wrapper">
            <iframe
                src="https://drive.google.com/file/d/1UFB0UXHQWMhlur0ezTkuJbZ9iRSwc0ts/preview"
                allow="autoplay"
                allowfullscreen>
            </iframe>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <style>
        /* Button Container */
        .button-container {
            position: fixed;
            bottom: 2rem;
            left: 50%;
            transform: translateX(-50%);
            z-index: 9999;
            padding: 10px;
        }

        /* Button Styling */
        .get-started-button {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 12px 35px;
            background: linear-gradient(90deg, 
                rgb(45, 211, 255) 0%, 
                rgb(88, 156, 255) 100%
            );
            text-decoration: none;
            border-radius: 50px;
            font-weight: 700;
            font-size: 18px;
            letter-spacing: 0.5px;
            transition: all 0.3s ease;
            box-shadow: 
                0 4px 15px rgba(45, 211, 255, 0.3),
                0 0 30px rgba(88, 156, 255, 0.2);
            border: none;
            cursor: pointer;
        }

        /* Text Styling */
        .button-text {
            background: linear-gradient(90deg, 
                #E0F7FF 0%, 
                #FFFFFF 100%
            );
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        .get-started-button:hover {
            transform: translateY(-2px);
            box-shadow: 
                0 8px 25px rgba(45, 211, 255, 0.4),
                0 0 50px rgba(88, 156, 255, 0.3);
        }

        /* Rocket emoji styling */
        .button-icon {
            font-size: 20px;
            margin-right: 2px;
            filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.1));
        }
        </style>

        <div class="button-container">
            <a href="https://huggingface.co/spaces/Sameer747/XAISupplyChainStreamlit" 
            target="_blank"
            class="get-started-button">
                <span class="button-icon">🚀</span>
                <span class="button-text">Get Started</span>
            </a>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()