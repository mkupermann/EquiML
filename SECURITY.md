# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   |  Yes            |
| 0.1.x   |  Security fixes only |
| < 0.1   |  No             |

## Reporting a Vulnerability

We take security seriously at EquiML. If you discover a security vulnerability, please follow these steps:

###  Private Disclosure

**DO NOT** create a public GitHub issue for security vulnerabilities.

Instead, please email security findings to:
**mkupermann@kupermann.com** with subject line "SECURITY: [Brief Description]"

###  What to Include

Please include the following information:
- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Suggested fix (if you have one)
- Your contact information

### ⏱ Response Timeline

- **Initial Response**: Within 24 hours
- **Assessment**: Within 72 hours
- **Fix Development**: 1-2 weeks (depending on severity)
- **Public Disclosure**: After fix is released

### 🏆 Recognition

We believe in recognizing security researchers who help make EquiML safer:
- Public acknowledgment (with your permission)
- Listing in our security hall of fame
- Potential bug bounty (for significant findings)

## Security Best Practices

###  For Users

When using EquiML:
- Always use the latest version
- Validate data inputs before processing
- Use secure environments for sensitive data
- Follow principle of least privilege
- Regularly update dependencies

### 🔐 For Contributors

When contributing to EquiML:
- Never commit secrets, API keys, or credentials
- Use parameterized queries for database operations
- Validate and sanitize all inputs
- Follow secure coding practices
- Run security linters before committing

## Security Features

###  Built-in Protections

EquiML includes:
- Input validation and sanitization
- Secure data handling practices
- Privacy-preserving bias analysis
- Audit logging for compliance
- Safe model serialization

###  Vulnerability Management

We actively:
- Monitor dependencies for known vulnerabilities
- Run automated security scans
- Conduct regular security reviews
- Maintain security documentation
- Provide timely security updates

## Compliance

EquiML is designed to support:
- **GDPR**: Data privacy and protection
- **CCPA**: California privacy compliance
- **SOC 2**: Security and availability controls
- **ISO 27001**: Information security management
- **EU AI Act**: AI system transparency and accountability

## Contact

For security-related questions or concerns:
- **Security Email**: mkupermann@kupermann.com
- **General Contact**: GitHub Issues (for non-security matters)
- **Documentation**: See docs/guides/ for security guidance

---

*Security is a shared responsibility. Thank you for helping keep EquiML and its users safe.*