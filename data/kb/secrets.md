---
id: secrets
title: Secrets and cloud credentials
---
## Project and environment secrets
Store passwords and keys as secrets in **Project settings > Secrets**. Secrets are encrypted at rest and injected as environment variables only into the jobs that ask for them. Environment-scoped secrets, such as those for `production`, are available only to jobs that deploy to that environment.

## Masking
Brightwave CI masks secret values in job logs, replacing them with `***`. Masking is a safety net, not a guarantee: avoid printing secrets or writing them into artifacts.

## Deploying to cloud providers without long-lived keys
Use OpenID Connect (OIDC). Each job can request a short-lived identity token from Brightwave CI, and AWS, Azure or Google Cloud exchange it for temporary credentials through a trust policy you create. With OIDC you do not store any cloud access keys in Brightwave CI.

## Rotating secrets
Update a secret's value in the settings page; running jobs keep the old value and new jobs get the new one. Rotate any secret that may have appeared in a log or an artifact.
