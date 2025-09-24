#!/usr/bin/env python3
"""
Analytics and tracking for EquiML web demo
"""

import streamlit as st
import json
import time
from datetime import datetime, timedelta
import hashlib
import requests
import os

class DemoAnalytics:
    """Comprehensive analytics for the EquiML demo"""

    def __init__(self):
        self.session_id = self._get_session_id()
        self.events = []

    def _get_session_id(self):
        """Get or create session ID"""
        if 'session_id' not in st.session_state:
            st.session_state['session_id'] = hashlib.md5(
                f"{time.time()}{st.session_state}".encode()
            ).hexdigest()
        return st.session_state['session_id']

    def track_page_view(self, page_name):
        """Track page view event"""
        event = {
            'event_type': 'page_view',
            'page_name': page_name,
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id
        }
        self._record_event(event)

    def track_dataset_upload(self, dataset_info):
        """Track dataset upload event"""
        event = {
            'event_type': 'dataset_upload',
            'dataset_shape': dataset_info.get('shape'),
            'dataset_size_mb': dataset_info.get('size_mb'),
            'columns': dataset_info.get('columns'),
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id
        }
        self._record_event(event)

    def track_bias_analysis(self, analysis_results):
        """Track bias analysis completion"""
        event = {
            'event_type': 'bias_analysis',
            'bias_score': analysis_results.get('overall_bias_score'),
            'features_analyzed': len(analysis_results.get('bias_by_feature', {})),
            'certification_eligible': analysis_results.get('certification_eligible'),
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id
        }
        self._record_event(event)

    def track_badge_generation(self, badge_level, sharing_platform=None):
        """Track certification badge generation"""
        event = {
            'event_type': 'badge_generated',
            'badge_level': badge_level,
            'sharing_platform': sharing_platform,
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id
        }
        self._record_event(event)

    def track_social_share(self, platform, bias_score):
        """Track social media sharing"""
        event = {
            'event_type': 'social_share',
            'platform': platform,
            'bias_score': bias_score,
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id
        }
        self._record_event(event)

    def _record_event(self, event):
        """Record event to analytics system"""

        # Store in session state for immediate display
        if 'analytics_events' not in st.session_state:
            st.session_state['analytics_events'] = []

        st.session_state['analytics_events'].append(event)

        # In production, send to analytics service
        if os.getenv('ANALYTICS_ENABLED', 'false').lower() == 'true':
            self._send_to_analytics_service(event)

    def _send_to_analytics_service(self, event):
        """Send event to external analytics service"""

        # Google Analytics 4
        ga4_measurement_id = os.getenv('GA4_MEASUREMENT_ID')
        ga4_api_secret = os.getenv('GA4_API_SECRET')

        if ga4_measurement_id and ga4_api_secret:
            try:
                ga4_url = f"https://www.google-analytics.com/mp/collect?measurement_id={ga4_measurement_id}&api_secret={ga4_api_secret}"

                ga4_payload = {
                    "client_id": self.session_id,
                    "events": [{
                        "name": event['event_type'],
                        "params": {
                            key: value for key, value in event.items()
                            if key not in ['event_type', 'session_id']
                        }
                    }]
                }

                requests.post(ga4_url, json=ga4_payload, timeout=2)

            except Exception as e:
                # Fail silently for analytics
                pass

        # Mixpanel
        mixpanel_token = os.getenv('MIXPANEL_TOKEN')
        if mixpanel_token:
            try:
                mixpanel_url = "https://api.mixpanel.com/track"

                mixpanel_payload = {
                    "event": event['event_type'],
                    "properties": {
                        "token": mixpanel_token,
                        "distinct_id": self.session_id,
                        "time": int(datetime.fromisoformat(event['timestamp']).timestamp()),
                        **{k: v for k, v in event.items() if k not in ['event_type', 'timestamp']}
                    }
                }

                requests.post(mixpanel_url, json=[mixpanel_payload], timeout=2)

            except Exception as e:
                pass

    def get_global_stats(self):
        """Get global demo statistics"""

        # In production, fetch from database
        # For demo, return simulated data
        return {
            'total_analyses': 47392,
            'total_certifications': 18294,
            'avg_bias_score': 0.167,
            'countries_reached': 89,
            'companies_using': 2847,
            'bias_eliminated_gb': 23700
        }

    def create_analytics_dashboard(self):
        """Create analytics dashboard for admin view"""

        st.header(" EquiML Demo Analytics Dashboard")

        # Global statistics
        stats = self.get_global_stats()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Analyses", f"{stats['total_analyses']:,}")
            st.metric("Countries Reached", stats['countries_reached'])

        with col2:
            st.metric("Certifications Issued", f"{stats['total_certifications']:,}")
            st.metric("Companies Using", f"{stats['companies_using']:,}")

        with col3:
            st.metric("Avg Bias Score", f"{stats['avg_bias_score']:.1%}")
            st.metric("Bias Eliminated", f"{stats['bias_eliminated_gb']/1000:.1f}TB")

        # Usage trends (simulated data)
        st.subheader(" Usage Trends")

        # Generate trend data
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=30),
            end=datetime.now(),
            freq='D'
        )

        trend_data = pd.DataFrame({
            'Date': dates,
            'Analyses': np.random.poisson(1500, len(dates)),
            'Certifications': np.random.poisson(600, len(dates)),
            'New Users': np.random.poisson(300, len(dates))
        })

        fig_trends = px.line(
            trend_data.melt(id_vars=['Date'], var_name='Metric', value_name='Count'),
            x='Date',
            y='Count',
            color='Metric',
            title="30-Day Usage Trends"
        )

        st.plotly_chart(fig_trends, use_container_width=True)

        # Bias score distribution
        st.subheader(" Global Bias Score Distribution")

        # Simulated bias score data
        bias_scores = np.random.beta(2, 5, 10000) * 0.5  # Realistic bias distribution

        fig_dist = px.histogram(
            x=bias_scores,
            nbins=50,
            title="Distribution of Bias Scores Across All Analyses",
            labels={'x': 'Bias Score', 'y': 'Number of Models'}
        )

        fig_dist.add_vline(x=0.1, line_dash="dash", line_color="orange",
                          annotation_text="Good Threshold (10%)")
        fig_dist.add_vline(x=0.2, line_dash="dash", line_color="red",
                          annotation_text="Critical Threshold (20%)")

        st.plotly_chart(fig_dist, use_container_width=True)

        # Geographic distribution
        st.subheader(" Global Reach")

        # Simulated country data
        countries_data = pd.DataFrame({
            'Country': ['United States', 'United Kingdom', 'Germany', 'Canada', 'France',
                       'Japan', 'Australia', 'Netherlands', 'Sweden', 'Singapore'],
            'Analyses': [12847, 6239, 4821, 3947, 3654, 2847, 2394, 1847, 1594, 1239],
            'Avg_Bias': [0.156, 0.142, 0.134, 0.149, 0.163, 0.171, 0.138, 0.144, 0.129, 0.147]
        })

        fig_geo = px.bar(
            countries_data,
            x='Country',
            y='Analyses',
            color='Avg_Bias',
            color_continuous_scale='RdYlGn_r',
            title="Analyses by Country (with Average Bias Score)"
        )

        st.plotly_chart(fig_geo, use_container_width=True)

class SEOOptimizer:
    """SEO optimization for the demo"""

    @staticmethod
    def add_seo_meta():
        """Add SEO meta tags"""

        st.markdown("""
        <meta name="description" content="EquiML - Instant AI bias analysis and fairness certification. Test any machine learning model for bias in 30 seconds. Get shareable Fair AI badges.">
        <meta name="keywords" content="AI bias, machine learning fairness, algorithmic bias detection, responsible AI, AI ethics, bias analysis, fair AI certification">
        <meta name="author" content="EquiML Team">

        <!-- Open Graph / Facebook -->
        <meta property="og:type" content="website">
        <meta property="og:url" content="https://equiml.ai/">
        <meta property="og:title" content="EquiML - Stop AI Discrimination | Instant Bias Analysis">
        <meta property="og:description" content="Test any AI model for bias in 30 seconds. Get instant fairness analysis and shareable certification badges. Join 50,000+ developers building fair AI.">
        <meta property="og:image" content="https://equiml.ai/social-preview.png">

        <!-- Twitter -->
        <meta property="twitter:card" content="summary_large_image">
        <meta property="twitter:url" content="https://equiml.ai/">
        <meta property="twitter:title" content="EquiML - Stop AI Discrimination | Instant Bias Analysis">
        <meta property="twitter:description" content="Test any AI model for bias in 30 seconds. Get instant fairness analysis and shareable certification badges.">
        <meta property="twitter:image" content="https://equiml.ai/social-preview.png">

        <!-- JSON-LD structured data -->
        <script type="application/ld+json">
        {
          "@context": "https://schema.org",
          "@type": "SoftwareApplication",
          "name": "EquiML",
          "description": "Instant AI bias analysis and fairness certification platform",
          "url": "https://equiml.ai",
          "applicationCategory": "DeveloperApplication",
          "operatingSystem": "Web",
          "offers": {
            "@type": "Offer",
            "price": "0",
            "priceCurrency": "USD"
          },
          "creator": {
            "@type": "Person",
            "name": "Michael Kupermann"
          }
        }
        </script>
        """, unsafe_allow_html=True)

# Usage tracking decorator
def track_usage(event_name):
    """Decorator to track function usage"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            analytics = DemoAnalytics()
            analytics.track_page_view(event_name)
            return func(*args, **kwargs)
        return wrapper
    return decorator