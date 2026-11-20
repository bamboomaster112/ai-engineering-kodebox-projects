---
id: rate-limits-status
title: API rate limits and service status
---
## Rate limits
The Brightwave CI API allows 5,000 requests per hour for each token. When you go over, the API answers with HTTP 429 and a `Retry-After` header that says how many seconds to wait. Clients should back off and retry after that time.

## Service status
Live service health and incident updates are published at status.brightwave.example. You can subscribe to email or webhook notifications there.
