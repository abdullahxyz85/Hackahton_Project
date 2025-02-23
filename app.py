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

def landing_page():
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
    }
    .feature-container {
        display: flex;
        justify-content: center;
        gap: 20px;
        flex-wrap: wrap;
        margin: 2rem 0;
    }
    .feature-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        flex: 1;
        min-width: 250px;
    }
    .cta-button {
        text-align: center;
        margin: 2rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header Section
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Welcome to SyncChain</h1>
        <p style='font-size: 1.2em; color: #666;'>
            AI-Powered Supply Chain Analytics Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📈 Demand Forecasting</h3>
            <p>AI-driven demand predictions with up to 95% accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>📦 Inventory Optimization</h3>
            <p>Smart inventory management with real-time insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>🚨 Risk Monitoring</h3>
            <p>Proactive risk detection and mitigation strategies</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-card">
            <h3>🎮 Scenario Planning</h3>
            <p>Interactive what-if analysis and simulation</p>
        </div>
        """, unsafe_allow_html=True)

    # Benefits Section
    st.markdown("---")
    st.markdown("<h2 style='text-align: center;'>Why Choose SyncChain?</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🤖 AI-Powered")
        st.write("Leveraging IBM Granite for intelligent supply chain insights")
    
    with col2:
        st.markdown("### 📊 Data-Driven")
        st.write("Make informed decisions with real-time analytics")
    
    with col3:
        st.markdown("### 💡 User-Friendly")
        st.write("Intuitive interface for seamless experience")

    # Call to Action
    st.markdown("---")
    if st.button("🚀 Get Started with SyncChain", key="start_button", use_container_width=True):
        st.session_state.show_main_app = True

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
            st.experimental_rerun()
            
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
        
        # Tabs Interface
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Demand Forecast", 
            "📦 Inventory Optimizer", 
            "🚨 Risk Monitor", 
            "🎮 Scenario Lab"
        ])

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

if __name__ == "__main__":
    main()
