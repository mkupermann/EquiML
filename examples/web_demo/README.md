# EquiML Live Web Demo

**Production-ready web demo for equiml.ai - Instant AI bias analysis with shareable certifications**

## Quick Start

### Local Development
```bash
cd web_demo
pip install -r requirements.txt
streamlit run app.py
```

### Production Deployment
```bash
cd web_demo
chmod +x deploy.sh
./deploy.sh
```

## Features

### **Instant Bias Analysis**
- Upload any CSV dataset
- Get bias analysis in seconds
- Real-time fairness scoring
- Comprehensive recommendations

### **Real-Time Visualizations**
- Interactive bias gauge
- Group outcome comparisons
- Fairness metrics dashboard
- Professional charts and graphs

### **Certified Fair AI Badges**
- Gold/Silver/Bronze certification levels
- Shareable HTML badges
- Social media integration
- Public verification system

### 📤 **Social Sharing**
- One-click Twitter sharing
- LinkedIn post templates
- Embeddable badges
- Viral marketing features

### 📈 **Live Analytics**
- Global usage statistics
- Real-time bias trends
- Community activity feed
- Performance tracking

## Architecture

```
web_demo/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Multi-service deployment
├── nginx.conf            # Reverse proxy configuration
├── deploy.sh             # Automated deployment script
└── README.md             # This file
```

## Deployment Options

### **Option 1: Streamlit Cloud (Easiest)**
1. Fork the EquiML repository
2. Connect to Streamlit Cloud
3. Deploy from `web_demo/app.py`
4. Custom domain: point equiml.ai to Streamlit

### **Option 2: Docker + Cloud (Recommended)**
1. Deploy to AWS/GCP/Azure
2. Use docker-compose.yml
3. Add SSL certificates
4. Configure DNS

### **Option 3: Kubernetes (Scale)**
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: equiml-demo
spec:
  replicas: 3
  selector:
    matchLabels:
      app: equiml-demo
  template:
    metadata:
      labels:
        app: equiml-demo
    spec:
      containers:
      - name: equiml-demo
        image: equiml/demo:latest
        ports:
        - containerPort: 8501
```

## Configuration

### **Environment Variables**
```bash
# .env
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ENABLE_CORS=false
STREAMLIT_THEME_PRIMARY_COLOR="#667eea"
ANALYTICS_ENABLED=true
DOMAIN_NAME=equiml.ai
```

### **SSL Setup**
```bash
# For Let's Encrypt (production)
certbot certonly --nginx -d equiml.ai -d www.equiml.ai

# Update nginx.conf with real certificates
ssl_certificate /etc/letsencrypt/live/equiml.ai/fullchain.pem;
ssl_certificate_key /etc/letsencrypt/live/equiml.ai/privkey.pem;
```

## Marketing Features

### **Viral Elements**
- **Instant Results**: Analysis in under 10 seconds
- **Shareable Badges**: Beautiful, embeddable certifications
- **Social Integration**: One-click sharing to Twitter/LinkedIn
- **Competitive Element**: Public fairness scoreboard

### **Professional Features**
- **Enterprise Ready**: Production deployment configuration
- **Analytics**: Comprehensive usage tracking
- **Certification**: Verifiable fair AI badges
- **Community**: Global statistics and trends

## Monitoring & Analytics

### **Built-in Analytics**
- Real-time usage statistics
- Bias score distributions
- Certification rates
- Geographic usage patterns

### **External Integration**
```python
# Google Analytics 4
GOOGLE_ANALYTICS_ID = "G-XXXXXXXXXX"

# Mixpanel for advanced analytics
MIXPANEL_TOKEN = "your_mixpanel_token"

# PostHog for user behavior
POSTHOG_API_KEY = "your_posthog_key"
```

## Security & Privacy

### **Data Privacy**
- No user data stored permanently
- Session-based analysis only
- GDPR compliant processing
- Optional analytics opt-out

### **Security Features**
- Rate limiting per IP
- Input sanitization
- XSS protection
- CSRF protection via Streamlit

## Performance Optimization

### **Caching Strategy**
```python
@st.cache_data(ttl=3600)
def analyze_bias_cached(df_hash, target_col, sensitive_cols):
    # Cached bias analysis for repeated datasets
    pass

@st.cache_resource
def load_models():
    # Cache model loading for performance
    pass
```

### **Load Testing**
```bash
# Test with Apache Bench
ab -n 1000 -c 10 http://localhost:8501/

# Test with artillery
artillery quick --count 100 --num 10 http://localhost:8501/
```

## Launch Strategy

### **Pre-Launch (1 week)**
1. Deploy to staging environment
2. Test with sample datasets
3. Verify all sharing features
4. Performance optimization

### **Launch Day**
1. Deploy to production
2. Submit to Product Hunt
3. Social media campaign
4. Influencer outreach

### **Post-Launch (ongoing)**
1. Monitor usage and performance
2. Collect user feedback
3. Iterate on features
4. Scale infrastructure

## Success Metrics

### **Technical KPIs**
- **Uptime**: >99.9%
- **Response Time**: <5 seconds for bias analysis
- **Concurrent Users**: 1,000+ simultaneous users
- **Error Rate**: <0.1%

### **Business KPIs**
- **Daily Active Users**: 1,000+
- **Analyses Performed**: 10,000+/month
- **Certification Badges**: 1,000+/month
- **Social Shares**: 500+/month

### **Growth KPIs**
- **GitHub Stars**: 10,000+ (from demo traffic)
- **Newsletter Signups**: 5,000+
- **Enterprise Inquiries**: 100+/month
- **Academic Partnerships**: 50+

## Support & Maintenance

### **Monitoring**
```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs -f

# Monitor resource usage
docker stats
```

### **Updates**
```bash
# Update application
git pull origin main
docker-compose build
docker-compose up -d
```

### **Backup**
```bash
# Backup analytics data (if stored locally)
docker exec equiml-demo tar czf /backup/analytics-$(date +%Y%m%d).tar.gz /app/data
```

---

**This web demo is designed to be the gateway that introduces the global AI community to EquiML's capabilities, making bias detection and fairness accessible to everyone while driving adoption through viral sharing and certification features.**
