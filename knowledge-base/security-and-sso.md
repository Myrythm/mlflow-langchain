---
title: Security and SSO
category: security
---

# Security and SSO

Nimbus is built with security as a foundation. This article covers encryption, two-factor authentication, single sign-on (SSO), and compliance certifications.

## Data Encryption

- **At rest:** All data stored by Nimbus is encrypted using **AES-256**.
- **In transit:** All communication between your browser or API client and Nimbus servers is encrypted using **TLS 1.2 or higher**.

## Two-Factor Authentication (2FA)

Nimbus supports **TOTP-based 2FA** (Time-based One-Time Passwords) for all plans and all users.

To enable 2FA on your account:

1. Go to **Settings → Security**.
2. Click **Enable two-factor authentication**.
3. Scan the QR code with an authenticator app (e.g., Google Authenticator, Authy, 1Password).
4. Enter the 6-digit code from your authenticator app to confirm.
5. Save your backup codes in a secure location.

Once enabled, you will be prompted for a TOTP code at every login.

Workspace Admins and Owners on Business and Enterprise plans can enforce 2FA for all members from **Settings → Security → Require 2FA**.

## Single Sign-On (SSO)

### Google and Microsoft SSO (Business and Enterprise)

Business and Enterprise workspaces can configure SSO with Google Workspace or Microsoft Entra ID (formerly Azure AD), allowing members to sign in with their corporate credentials.

To configure Google or Microsoft SSO:

1. Navigate to **Settings → Security → Single Sign-On**.
2. Select **Google** or **Microsoft** as your identity provider (IdP).
3. Follow the on-screen instructions to authorize Nimbus in your IdP admin console.
4. Once configured, members are redirected to your IdP's login page when they access Nimbus.

### SAML SSO and SCIM Provisioning (Enterprise)

Enterprise workspaces support any SAML 2.0-compliant identity provider, including Okta, OneLogin, Ping Identity, and Microsoft Entra ID.

**Configuring SAML SSO:**

1. Navigate to **Settings → Security → SAML SSO**.
2. Copy the **Assertion Consumer Service (ACS) URL** and **Entity ID** from Nimbus.
3. In your IdP, create a new SAML application and paste the ACS URL and Entity ID.
4. Download the IdP metadata XML (or copy the metadata URL) and paste it into the Nimbus SAML configuration screen.
5. Click **Test connection** to verify the setup, then click **Enable SAML SSO**.

**SCIM Provisioning:**

SCIM (System for Cross-domain Identity Management) automates user provisioning and deprovisioning from your IdP to Nimbus.

1. Navigate to **Settings → Security → SCIM**.
2. Copy the **SCIM base URL** and generate a **SCIM API token**.
3. Enter these values in your IdP's SCIM provisioning settings.
4. Assign users and groups to the Nimbus application in your IdP. Nimbus creates, updates, and deactivates member accounts automatically.

## Audit Logs (Enterprise)

Enterprise workspaces have access to a full audit log of security-relevant events — logins, member changes, permission changes, SSO configuration updates, and API key usage.

Access audit logs via **Settings → Security → Audit Logs**.

## Data Residency (Enterprise)

Enterprise customers can choose where their data is stored:

- **United States (US)**
- **European Union (EU)**

Data residency is configured during Enterprise onboarding. Contact your dedicated customer success manager to change your residency region.

## Compliance

Nimbus holds a **SOC 2 Type II** certification, audited annually by an independent third party. A copy of the audit report is available to Enterprise customers under NDA. Contact your customer success manager to request it.

## Uptime SLA

Enterprise workspaces are covered by a **99.9% uptime SLA**. For SLA credits or uptime history, contact your customer success manager.

## Reporting a Security Issue

If you discover a potential security vulnerability in Nimbus, please report it responsibly by emailing security@nimbus.io. Do not disclose vulnerabilities publicly before the Nimbus security team has had a chance to investigate.
