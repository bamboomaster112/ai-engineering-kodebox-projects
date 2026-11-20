---
id: api-tokens
title: Personal access tokens for the API
---
## Creating a token
Create personal access tokens under **Account > Tokens**. Tokens start with the prefix `bw_pat_`. Choose only the scopes you need, such as `pipelines:read` or `pipelines:write`, and an expiry date. Tokens can last at most one year.

## Rotating a token
To rotate a token, create a new token with the same scopes, update every script or integration that uses the old one, and then revoke the old token. Brightwave CI does not change a token's value in place.

## Revoking a token
Revoke a token from the same page. Revocation is immediate: requests with that token fail with HTTP 401. If a token was exposed, revoke it first and investigate afterwards.
