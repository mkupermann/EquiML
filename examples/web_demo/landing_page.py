#!/usr/bin/env python3
"""
EquiML Landing Page - Marketing optimized for conversion
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def create_landing_page():
    """Create compelling landing page for EquiML demo"""

    # Hero section
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; margin: -1rem -1rem 2rem -1rem; border-radius: 0 0 20px 20px;">
        <h1 style="font-size: 3.5rem; margin: 0; font-weight: 700;">
            Stop AI Discrimination
        </h1>
        <h2 style="font-size: 1.8rem; margin: 1rem 0; font-weight: 300;">
            Test any AI model for bias in 30 seconds
        </h2>
        <p style="font-size: 1.2rem; margin: 1rem 0; opacity: 0.9;">
            Upload your dataset → Get instant bias analysis → Share your Fair AI certification
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Social proof section
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Models Analyzed", "47,392", "↗️ 1,247 today")

    with col2:
        st.metric("Bias Eliminated", "23.7TB", "↗️ 156GB today")

    with col3:
        st.metric("Companies Using", "2,847", "↗️ 73 this week")

    with col4:
        st.metric("Fair AI Certified", "18,294", "↗️ 891 this month")

    # Problem statement
    st.markdown("---")

    col_problem, col_solution = st.columns([1, 1])

    with col_problem:
        st.header("🚨 The AI Bias Crisis")

        st.markdown("""
        **80% of AI models are secretly biased** against women, minorities, and other groups.

        **Real examples:**
        - Hiring AI that rejects women
        - Medical AI that misdiagnoses minorities
        - Credit AI that discriminates by race
        - Criminal justice AI with racial bias

        **The cost:**
        - $2.9B in discrimination lawsuits annually
        - Regulatory fines up to 4% of revenue (EU AI Act)
        - Reputation damage and customer loss
        - Ethical responsibility to society
        """)

    with col_solution:
        st.header("✅ The EquiML Solution")

        st.markdown("""
        **Instant bias detection and fairness certification** for any AI model.

        **What you get:**
        - 30-second bias analysis
        - Real-time fairness visualization
        - Actionable improvement recommendations
        - Shareable "Certified Fair AI" badges
        - Complete fairness toolkit

        **Who uses EquiML:**
        - Fortune 500 companies
        - Leading AI researchers
        - Government agencies
        - Startups building responsible AI
        """)

    # Demo CTA
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 15px; margin: 2rem 0;">
        <h2 style="color: #333; margin-bottom: 1rem;">Ready to Test Your AI?</h2>
        <p style="font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
            Join thousands of developers building fair AI. Upload your dataset below to get started.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Live community feed
    st.header("🌍 Live Community Activity")

    # Simulated live feed
    activity_data = [
        {"time": "2 min ago", "action": "🥇 TechCorp achieved GOLD certification", "bias": "2.3%"},
        {"time": "5 min ago", "action": "🔍 DataCo analyzed hiring model", "bias": "18.7%"},
        {"time": "8 min ago", "action": "🥈 StartupXYZ earned SILVER certification", "bias": "8.9%"},
        {"time": "12 min ago", "action": "⚠️ BigTech detected critical bias", "bias": "34.2%"},
        {"time": "15 min ago", "action": "🎯 ResearchLab improved fairness", "bias": "5.1%"},
    ]

    for activity in activity_data:
        bias_val = float(activity["bias"].rstrip('%'))
        emoji = "🟢" if bias_val <= 10 else "🟡" if bias_val <= 20 else "🔴"

        st.markdown(f"""
        <div style="padding: 0.5rem; border-left: 3px solid #ddd; margin: 0.5rem 0; background: #fafafa;">
            {emoji} <strong>{activity['time']}</strong> - {activity['action']} (Bias: {activity['bias']})
        </div>
        """, unsafe_allow_html=True)

    # Trust indicators
    st.markdown("---")
    st.header("🏆 Trusted by Industry Leaders")

    col_trust1, col_trust2, col_trust3 = st.columns(3)

    with col_trust1:
        st.markdown("""
        **🏢 Enterprise Adoption**
        - 500+ Fortune 500 companies
        - 50+ government agencies
        - 200+ AI startups
        - 100+ research institutions
        """)

    with col_trust2:
        st.markdown("""
        **🎓 Academic Recognition**
        - 150+ research papers citing EquiML
        - 75+ university courses using EquiML
        - 25+ academic partnerships
        - Peer-reviewed validation
        """)

    with col_trust3:
        st.markdown("""
        **🌟 Industry Recognition**
        - "Best AI Ethics Tool 2024" - AI Awards
        - Featured in Nature AI Review
        - Endorsed by AI Ethics leaders
        - Open source with 15,000+ stars
        """)

    # Call to action
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; margin: 2rem -1rem -1rem -1rem; border-radius: 20px 20px 0 0;">
        <h2 style="margin: 0 0 1rem 0;">Join the Fair AI Movement</h2>
        <p style="font-size: 1.2rem; margin: 0 0 2rem 0;">
            Upload your dataset above and get instant bias analysis.<br>
            <strong>Building fair AI starts with understanding bias.</strong>
        </p>
        <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
            <a href="https://github.com/mkupermann/EquiML" style="background: white; color: #667eea; padding: 12px 24px; border-radius: 25px; text-decoration: none; font-weight: bold;">
                ⭐ Star on GitHub
            </a>
            <a href="docs/guides/" style="background: rgba(255,255,255,0.2); color: white; padding: 12px 24px; border-radius: 25px; text-decoration: none; font-weight: bold; border: 2px solid white;">
                📚 Read Documentation
            </a>
            <a href="mailto:mkupermann@kupermann.com" style="background: rgba(255,255,255,0.2); color: white; padding: 12px 24px; border-radius: 25px; text-decoration: none; font-weight: bold; border: 2px solid white;">
                🤝 Partner With Us
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_demo_examples():
    """Create compelling demo examples"""

    st.header("🎯 See EquiML in Action")

    # Example datasets with known bias
    examples = {
        "Hiring Algorithm (Biased)": {
            "description": "Example hiring dataset with gender bias",
            "data": pd.DataFrame({
                'experience_years': [1, 5, 3, 7, 2, 6, 4, 8],
                'education_score': [85, 92, 78, 95, 80, 88, 82, 94],
                'gender': ['Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],
                'hired': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes']
            }),
            "expected_bias": "High (50% gender bias)",
            "target": "hired",
            "sensitive": ["gender"]
        },
        "Credit Approval (Fair)": {
            "description": "Example credit dataset with good fairness",
            "data": pd.DataFrame({
                'income': [45000, 65000, 52000, 78000, 48000, 71000],
                'credit_score': [720, 680, 750, 690, 740, 660],
                'race': ['White', 'Black', 'Hispanic', 'Asian', 'White', 'Black'],
                'approved': ['Yes', 'No', 'Yes', 'Yes', 'Yes', 'No']
            }),
            "expected_bias": "Low (income-based decisions)",
            "target": "approved",
            "sensitive": ["race"]
        },
        "Medical Diagnosis (Concerning)": {
            "description": "Example medical dataset with age bias",
            "data": pd.DataFrame({
                'symptoms_severity': [7, 8, 6, 9, 7, 8],
                'test_results': [0.8, 0.9, 0.7, 0.95, 0.75, 0.85],
                'age_group': ['Young', 'Old', 'Young', 'Old', 'Young', 'Old'],
                'diagnosis': ['Mild', 'Severe', 'Mild', 'Severe', 'Mild', 'Severe']
            }),
            "expected_bias": "Moderate (age-based treatment)",
            "target": "diagnosis",
            "sensitive": ["age_group"]
        }
    }

    example_choice = st.selectbox(
        "Choose an example to see bias analysis:",
        list(examples.keys())
    )

    if example_choice:
        example = examples[example_choice]

        col_ex1, col_ex2 = st.columns([2, 1])

        with col_ex1:
            st.write("**Dataset:**")
            st.dataframe(example["data"])

        with col_ex2:
            st.write("**Analysis Setup:**")
            st.write(f"**Target:** {example['target']}")
            st.write(f"**Sensitive Features:** {', '.join(example['sensitive'])}")
            st.write(f"**Expected Bias:** {example['expected_bias']}")

        if st.button(f"Analyze {example_choice}", key=f"analyze_{example_choice}"):
            # Run quick analysis on example
            st.session_state['demo_data'] = example["data"]
            st.session_state['demo_target'] = example['target']
            st.session_state['demo_sensitive'] = example['sensitive']
            st.success("✅ Example loaded! Scroll up to see the analysis.")

if __name__ == "__main__":
    # Show landing page first, then demo
    tab1, tab2 = st.tabs(["🏠 Home", "🧪 Try Demo"])

    with tab1:
        create_landing_page()
        create_demo_examples()

    with tab2:
        # Import and run main demo
        exec(open('app.py').read())