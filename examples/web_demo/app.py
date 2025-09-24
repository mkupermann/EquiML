#!/usr/bin/env python3
"""
EquiML Live Web Demo - equiml.ai
Production-ready Streamlit app for instant bias analysis and fairness visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import hashlib
import json
import os
import sys
from datetime import datetime
import uuid

# Add EquiML to path
sys.path.append('..')
from src.data import Data
from src.model import Model
from src.evaluation import EquiMLEvaluation
from src.monitoring import BiasMonitor

# Page configuration
st.set_page_config(
    page_title="EquiML - Instant AI Bias Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }

    .bias-score-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }

    .bias-score-medium {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }

    .bias-score-low {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }

    .certified-badge {
        background: linear-gradient(45deg, #4caf50, #45a049);
        color: white;
        padding: 10px 20px;
        border-radius: 25px;
        text-align: center;
        font-weight: bold;
        margin: 20px 0;
    }

    .demo-stats {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

class InstantBiasAnalyzer:
    """Ultra-fast bias analysis for web demo"""

    def __init__(self):
        self.analysis_cache = {}

    def quick_bias_scan(self, df, target_col, sensitive_cols):
        """Lightning-fast bias analysis for web demo"""

        # Create cache key
        cache_key = hashlib.md5(f"{df.shape}{target_col}{sensitive_cols}".encode()).hexdigest()

        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]

        results = {
            'overall_bias_score': 0.0,
            'bias_by_feature': {},
            'fairness_metrics': {},
            'recommendations': [],
            'certification_eligible': False
        }

        try:
            # Quick statistical bias detection
            for sensitive_col in sensitive_cols:
                if sensitive_col in df.columns and target_col in df.columns:

                    # Calculate group-based outcome rates
                    group_rates = {}
                    for group in df[sensitive_col].unique():
                        group_data = df[df[sensitive_col] == group]
                        if len(group_data) > 0:
                            if df[target_col].dtype == 'object':
                                # For categorical targets, calculate positive rate
                                positive_outcomes = group_data[target_col].value_counts()
                                total = len(group_data)
                                if len(positive_outcomes) > 1:
                                    # Assume last value alphabetically is "positive"
                                    positive_rate = positive_outcomes.iloc[-1] / total
                                else:
                                    positive_rate = 0.5
                            else:
                                # For numerical targets, calculate mean
                                positive_rate = group_data[target_col].mean()

                            group_rates[str(group)] = positive_rate

                    # Calculate bias as max difference between groups
                    if len(group_rates) >= 2:
                        rates = list(group_rates.values())
                        bias_score = max(rates) - min(rates)
                        results['bias_by_feature'][sensitive_col] = {
                            'bias_score': bias_score,
                            'group_rates': group_rates,
                            'severity': 'HIGH' if bias_score > 0.2 else 'MEDIUM' if bias_score > 0.1 else 'LOW'
                        }

                        # Update overall bias score
                        results['overall_bias_score'] = max(results['overall_bias_score'], bias_score)

            # Generate quick recommendations
            if results['overall_bias_score'] > 0.2:
                results['recommendations'] = [
                    "CRITICAL: High bias detected - immediate action required",
                    "Apply EquiML's bias mitigation techniques",
                    "Use fairness-constrained training",
                    "Implement real-time bias monitoring"
                ]
            elif results['overall_bias_score'] > 0.1:
                results['recommendations'] = [
                    "MODERATE: Some bias detected - improvement recommended",
                    "Consider data rebalancing techniques",
                    "Apply fairness constraints during training"
                ]
            else:
                results['recommendations'] = [
                    "EXCELLENT: Low bias detected",
                    "Continue monitoring for bias drift",
                    "Consider EquiML certification"
                ]
                results['certification_eligible'] = True

            # Cache results
            self.analysis_cache[cache_key] = results

        except Exception as e:
            st.error(f"Analysis error: {str(e)}")

        return results

class FairnessVisualizer:
    """Real-time fairness visualization engine"""

    @staticmethod
    def create_bias_dashboard(bias_results, df):
        """Create comprehensive bias visualization dashboard"""

        if not bias_results['bias_by_feature']:
            st.warning("No bias analysis possible - please ensure you have selected both target and sensitive features")
            return

        # Overall bias gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = bias_results['overall_bias_score'] * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Overall Bias Score (%)"},
            delta = {'reference': 10},
            gauge = {
                'axis': {'range': [None, 50]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 10], 'color': "lightgreen"},
                    {'range': [10, 20], 'color': "yellow"},
                    {'range': [20, 50], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 20
                }
            }
        ))

        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Bias by feature
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Bias by Sensitive Feature")

            bias_data = []
            for feature, data in bias_results['bias_by_feature'].items():
                bias_data.append({
                    'Feature': feature,
                    'Bias Score': data['bias_score'] * 100,
                    'Severity': data['severity']
                })

            if bias_data:
                bias_df = pd.DataFrame(bias_data)

                # Color map for severity
                color_map = {'LOW': 'green', 'MEDIUM': 'orange', 'HIGH': 'red'}
                colors = [color_map[severity] for severity in bias_df['Severity']]

                fig_bar = px.bar(
                    bias_df,
                    x='Feature',
                    y='Bias Score',
                    color='Severity',
                    color_discrete_map=color_map,
                    title="Bias Score by Feature (%)"
                )
                fig_bar.add_hline(y=10, line_dash="dash", line_color="orange",
                                annotation_text="Acceptable Threshold (10%)")
                fig_bar.add_hline(y=20, line_dash="dash", line_color="red",
                                annotation_text="Critical Threshold (20%)")

                st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            st.subheader("Group Outcome Rates")

            # Create group comparison chart
            for feature, data in bias_results['bias_by_feature'].items():
                if 'group_rates' in data:
                    group_df = pd.DataFrame(
                        list(data['group_rates'].items()),
                        columns=['Group', 'Outcome Rate']
                    )
                    group_df['Outcome Rate'] = group_df['Outcome Rate'] * 100

                    fig_group = px.bar(
                        group_df,
                        x='Group',
                        y='Outcome Rate',
                        title=f"Outcome Rates by {feature}",
                        color='Outcome Rate',
                        color_continuous_scale='RdYlGn'
                    )
                    st.plotly_chart(fig_group, use_container_width=True)

class CertificationBadgeGenerator:
    """Generate shareable Certified Fair AI badges"""

    @staticmethod
    def generate_badge(bias_score, company_name="", model_name="", share_id=""):
        """Generate certification badge based on bias score"""

        if bias_score <= 0.1:
            badge_level = "GOLD"
            badge_color = "#FFD700"
            badge_text = "Certified Fair AI - Gold"
            description = "Exceptional fairness standards"
        elif bias_score <= 0.2:
            badge_level = "SILVER"
            badge_color = "#C0C0C0"
            badge_text = "Certified Fair AI - Silver"
            description = "Good fairness standards"
        elif bias_score <= 0.3:
            badge_level = "BRONZE"
            badge_color = "#CD7F32"
            badge_text = "Certified Fair AI - Bronze"
            description = "Basic fairness standards"
        else:
            badge_level = "NEEDS_IMPROVEMENT"
            badge_color = "#FF6B6B"
            badge_text = "Bias Detected - Improvement Needed"
            description = "Significant bias found"

        # Create badge HTML
        badge_html = f"""
        <div style="
            background: linear-gradient(135deg, {badge_color}20, {badge_color}40);
            border: 3px solid {badge_color};
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            margin: 20px 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        ">
            <h2 style="color: {badge_color}; margin: 0; font-size: 1.5rem;">
                {badge_text}
            </h2>
            <p style="margin: 10px 0; font-size: 1.1rem; color: #333;">
                {description}
            </p>
            <p style="margin: 5px 0; color: #666;">
                Bias Score: {bias_score:.1%} | Verified by EquiML
            </p>
            {f'<p style="margin: 5px 0; color: #666;">{company_name} - {model_name}</p>' if company_name else ''}
            <p style="margin: 5px 0; color: #999; font-size: 0.9rem;">
                Certified on {datetime.now().strftime('%Y-%m-%d')} | ID: {share_id[:8]}
            </p>
        </div>
        """

        return badge_html, badge_level

    @staticmethod
    def generate_shareable_link(results, share_id):
        """Generate shareable link for results"""

        # Create shareable URL (in production, save to database)
        base_url = "https://equiml.ai/results"
        share_url = f"{base_url}/{share_id}"

        # Social media sharing templates
        twitter_text = f"I just tested my AI model for bias using @EquiML! "

        if results['overall_bias_score'] <= 0.1:
            twitter_text += f" GOLD certification - {results['overall_bias_score']:.1%} bias detected. Building fair AI! #AIFairness #ResponsibleAI"
        elif results['overall_bias_score'] <= 0.2:
            twitter_text += f"🥈 SILVER certification - {results['overall_bias_score']:.1%} bias detected. Working on improvements! #AIFairness"
        else:
            twitter_text += f" {results['overall_bias_score']:.1%} bias detected. Time to make AI fairer! #AIBias #EquiML"

        linkedin_text = f"""
Just analyzed my AI model for bias using EquiML - the results are eye-opening!

 Bias Score: {results['overall_bias_score']:.1%}
 Status: {results['certification_eligible'] and 'Certified Fair AI' or 'Improvement Needed'}
 Analysis: Instant bias detection and fairness evaluation

Building responsible AI isn't just ethical - it's essential for business success and regulatory compliance.

Try it yourself: equiml.ai
#AIFairness #ResponsibleAI #MachineLearning #TechEthics
        """

        return {
            'share_url': share_url,
            'twitter_text': twitter_text,
            'linkedin_text': linkedin_text,
            'share_id': share_id
        }

class AnalyticsTracker:
    """Track usage analytics for the demo"""

    def __init__(self):
        self.session_id = st.session_state.get('session_id', str(uuid.uuid4()))
        st.session_state['session_id'] = self.session_id

    def track_analysis(self, dataset_info, bias_score, processing_time):
        """Track analysis event"""

        event = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'dataset_shape': dataset_info.get('shape'),
            'bias_score': bias_score,
            'processing_time': processing_time,
            'certification_eligible': bias_score <= 0.2
        }

        # In production, send to analytics service
        # For demo, store in session state
        if 'analytics_events' not in st.session_state:
            st.session_state['analytics_events'] = []

        st.session_state['analytics_events'].append(event)

    def get_demo_stats(self):
        """Get demo usage statistics"""

        events = st.session_state.get('analytics_events', [])

        if not events:
            return {
                'total_analyses': 0,
                'avg_bias_score': 0.0,
                'certification_rate': 0.0
            }

        total_analyses = len(events)
        avg_bias_score = np.mean([e['bias_score'] for e in events])
        certifications = sum(1 for e in events if e['certification_eligible'])
        certification_rate = certifications / total_analyses if total_analyses > 0 else 0

        return {
            'total_analyses': total_analyses,
            'avg_bias_score': avg_bias_score,
            'certification_rate': certification_rate,
            'avg_processing_time': np.mean([e['processing_time'] for e in events])
        }

def main():
    """Main EquiML web demo application"""

    # Initialize components
    analyzer = InstantBiasAnalyzer()
    visualizer = FairnessVisualizer()
    badge_generator = CertificationBadgeGenerator()
    analytics = AnalyticsTracker()

    # Header
    st.markdown('<h1 class="main-header">EquiML: Instant AI Bias Analysis</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; font-size: 1.2rem; color: #666;">
        Upload any dataset and get instant bias analysis with real-time fairness visualization.<br>
        <strong>Build AI that's fair for everyone.</strong>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for demo stats
    with st.sidebar:
        st.header(" Global Impact")

        # Demo statistics
        stats = analytics.get_demo_stats()

        st.markdown(f"""
        <div class="demo-stats">
            <h4>Live Demo Stats</h4>
            <p> Analyses Run: <strong>{stats['total_analyses']:,}</strong></p>
            <p> Avg Bias Score: <strong>{stats['avg_bias_score']:.1%}</strong></p>
            <p>🏆 Certification Rate: <strong>{stats['certification_rate']:.1%}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Sample datasets
        st.header("📁 Try Sample Datasets")

        if st.button("Load Adult Income Dataset"):
            st.session_state['use_sample'] = True
            st.rerun()

        if st.button("Load Credit Approval Dataset"):
            # Create sample credit dataset
            credit_data = pd.DataFrame({
                'income': np.random.normal(50000, 20000, 100),
                'credit_score': np.random.normal(650, 100, 100),
                'age': np.random.randint(18, 80, 100),
                'gender': np.random.choice(['Male', 'Female'], 100),
                'race': np.random.choice(['White', 'Black', 'Hispanic', 'Asian'], 100),
                'approved': np.random.choice(['Yes', 'No'], 100)
            })
            st.session_state['sample_data'] = credit_data
            st.session_state['use_sample'] = True
            st.rerun()

        st.markdown("---")

        st.header(" Quick Actions")
        st.markdown("""
        **🔗 Share EquiML:**
        - [GitHub](https://github.com/mkupermann/EquiML)
        - [Documentation](docs/guides/)
        - [Research Paper](coming-soon)

        **📧 Get Updates:**
        - Product launches
        - New features
        - Community events
        """)

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📤 Upload Your Dataset")

        df = None

        # Check for sample data
        if st.session_state.get('use_sample', False):
            if 'sample_data' in st.session_state:
                df = st.session_state['sample_data']
                st.success(" Sample dataset loaded!")
            else:
                # Load adult dataset
                try:
                    df = pd.read_csv('../tests/adult.csv')
                    st.success(" Adult Income dataset loaded!")
                except:
                    st.error("Sample dataset not found")

            if st.button("Clear Sample Data"):
                st.session_state['use_sample'] = False
                if 'sample_data' in st.session_state:
                    del st.session_state['sample_data']
                st.rerun()

        else:
            # File upload
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type="csv",
                help="Upload your dataset to analyze for bias. Supports CSV files up to 200MB."
            )

            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f" Dataset uploaded: {df.shape[0]} rows, {df.shape[1]} columns")
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")

        if df is not None:
            # Dataset preview
            with st.expander(" Dataset Preview", expanded=False):
                st.write("**Dataset Shape:**", df.shape)
                st.write("**First 5 rows:**")
                st.dataframe(df.head())

                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.write("**Data Types:**")
                    st.write(df.dtypes.value_counts())

                with col_info2:
                    st.write("**Missing Values:**")
                    missing = df.isnull().sum()
                    if missing.sum() > 0:
                        st.write(missing[missing > 0])
                    else:
                        st.write("No missing values ")

            # Feature selection
            st.header("⚙ Configure Analysis")

            col_config1, col_config2 = st.columns(2)

            with col_config1:
                target_column = st.selectbox(
                    " Select Target Column (what you're predicting)",
                    options=df.columns.tolist(),
                    help="This is the outcome you want your AI to predict fairly"
                )

            with col_config2:
                sensitive_features = st.multiselect(
                    " Select Sensitive Features (for fairness analysis)",
                    options=[col for col in df.columns if col != target_column],
                    default=[col for col in df.columns if any(keyword in col.lower()
                            for keyword in ['sex', 'gender', 'race', 'age', 'ethnicity'])],
                    help="These are characteristics that should NOT influence your AI's decisions"
                )

            # Analysis button
            if st.button(" Analyze for Bias", type="primary", use_container_width=True):
                if not sensitive_features:
                    st.warning(" Please select at least one sensitive feature for bias analysis")
                else:
                    # Run instant bias analysis
                    with st.spinner(" Analyzing dataset for bias..."):
                        start_time = time.time()

                        # Quick bias analysis
                        bias_results = analyzer.quick_bias_scan(df, target_column, sensitive_features)

                        processing_time = time.time() - start_time

                        # Track analytics
                        analytics.track_analysis(
                            {'shape': df.shape},
                            bias_results['overall_bias_score'],
                            processing_time
                        )

                        # Store results for sharing
                        share_id = str(uuid.uuid4())
                        st.session_state['current_results'] = {
                            'bias_results': bias_results,
                            'dataset_info': {
                                'shape': df.shape,
                                'target': target_column,
                                'sensitive_features': sensitive_features
                            },
                            'share_id': share_id,
                            'timestamp': datetime.now().isoformat()
                        }

                    st.success(f" Analysis completed in {processing_time:.2f} seconds!")
                    st.rerun()

    with col2:
        # Live statistics panel
        st.header(" Live Statistics")

        stats = analytics.get_demo_stats()

        # Statistics display
        st.metric("Total Analyses", f"{stats['total_analyses']:,}")
        st.metric("Avg Bias Score", f"{stats['avg_bias_score']:.1%}")
        st.metric("Certification Rate", f"{stats['certification_rate']:.1%}")

        # Recent activity
        if stats['total_analyses'] > 0:
            st.markdown("** Recent Activity:**")
            events = st.session_state.get('analytics_events', [])[-5:]
            for event in reversed(events):
                bias_level = "" if event['bias_score'] <= 0.1 else "" if event['bias_score'] <= 0.2 else ""
                st.markdown(f"{bias_level} {event['bias_score']:.1%} bias detected")

    # Results display
    if 'current_results' in st.session_state:
        results = st.session_state['current_results']
        bias_results = results['bias_results']

        st.markdown("---")
        st.header(" Analysis Results")

        # Overall bias score display
        bias_score = bias_results['overall_bias_score']

        if bias_score <= 0.1:
            st.markdown(f"""
            <div class="bias-score-low">
                <h3> Excellent Fairness Score!</h3>
                <p><strong>Overall Bias: {bias_score:.1%}</strong></p>
                <p>Your model shows excellent fairness characteristics. Consider applying for EquiML certification!</p>
            </div>
            """, unsafe_allow_html=True)
        elif bias_score <= 0.2:
            st.markdown(f"""
            <div class="bias-score-medium">
                <h3> Moderate Bias Detected</h3>
                <p><strong>Overall Bias: {bias_score:.1%}</strong></p>
                <p>Your model shows some bias. Consider applying EquiML's bias mitigation techniques.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="bias-score-high">
                <h3> High Bias Detected!</h3>
                <p><strong>Overall Bias: {bias_score:.1%}</strong></p>
                <p>Significant bias detected. Immediate action recommended to ensure fair AI.</p>
            </div>
            """, unsafe_allow_html=True)

        # Detailed visualizations
        st.header(" Detailed Fairness Analysis")
        visualizer.create_bias_dashboard(bias_results, df)

        # Recommendations
        st.header(" Actionable Recommendations")

        for i, rec in enumerate(bias_results['recommendations'], 1):
            if 'CRITICAL' in rec:
                st.error(f"{i}. {rec}")
            elif 'MODERATE' in rec:
                st.warning(f"{i}. {rec}")
            else:
                st.info(f"{i}. {rec}")

        # Certification badge
        st.header("🏆 Certification Badge")

        company_name = st.text_input("Company Name (optional)", placeholder="Your Company")
        model_name = st.text_input("Model Name (optional)", placeholder="Your AI Model")

        badge_html, badge_level = badge_generator.generate_badge(
            bias_score, company_name, model_name, results['share_id']
        )

        st.markdown(badge_html, unsafe_allow_html=True)

        # Sharing options
        st.header("📤 Share Your Results")

        share_info = badge_generator.generate_shareable_link(bias_results, results['share_id'])

        col_share1, col_share2 = st.columns(2)

        with col_share1:
            st.markdown("**🐦 Share on Twitter:**")
            twitter_url = f"https://twitter.com/intent/tweet?text={share_info['twitter_text']}"
            st.markdown(f"[Tweet Results]({twitter_url})")

            st.markdown("** Share on LinkedIn:**")
            st.text_area("LinkedIn Post", share_info['linkedin_text'], height=100)

        with col_share2:
            st.markdown("**🔗 Shareable Link:**")
            st.code(share_info['share_url'])

            st.markdown("** Embed Badge:**")
            st.code(badge_html, language='html')

        # Call to action
        st.markdown("---")
        st.header(" Take Action")

        col_action1, col_action2, col_action3 = st.columns(3)

        with col_action1:
            st.markdown("""
            **🛠 Fix the Bias**

            [Download EquiML](https://github.com/mkupermann/EquiML)

            Use our comprehensive toolkit to eliminate bias and build fair AI.
            """)

        with col_action2:
            st.markdown("""
            ** Learn More**

            [Read Our Guides](docs/guides/)

            Complete tutorials from beginner to advanced LLM development.
            """)

        with col_action3:
            st.markdown("""
            **🤝 Get Support**

            [Join Community](https://github.com/mkupermann/EquiML/discussions)

            Connect with other developers building fair AI.
            """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>Built with ❤ by the EquiML team | Making AI fair for everyone</p>
        <p>
            <a href="https://github.com/mkupermann/EquiML">GitHub</a> |
            <a href="docs/guides/">Documentation</a> |
            <a href="mailto:mkupermann@kupermann.com">Contact</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()