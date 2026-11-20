---
id: sso-saml
title: Single sign-on with SAML and SCIM
---
## Availability
SAML single sign-on and SCIM user provisioning are available on the Enterprise plan only. Free and Team organizations sign in with email and password or with their Git provider.

## Setting up SAML
An organization owner adds the identity provider's metadata URL under **Organization settings > Security**, tests the connection, and then turns on **Require SSO**. Once required, members must sign in through the identity provider.

## Recovery codes
Owners get ten recovery codes when SSO is enforced. Store them safely: they are the only way to sign in if the identity provider is unavailable.
