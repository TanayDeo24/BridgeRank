# Security policy

## Reporting a vulnerability

Please do not disclose suspected vulnerabilities in a public issue. Use GitHub's private vulnerability
reporting feature for this repository and include the affected component, reproduction steps, impact, and any
suggested mitigation. I will acknowledge a complete report as soon as practical, validate the finding, and
coordinate remediation before public disclosure.

## Security model

- The API can require an `X-API-Key` through `SEARCH_API_KEY`.
- Raw query retention is disabled by default; the event store persists a normalized SHA-256 query hash.
- The production container runs as a non-root user.
- Index artifacts are validated against SHA-256 checksums before readiness is enabled.
- Database credentials must be supplied through the deployment platform's secret manager.
- TLS, rate limiting, network policy, and identity-aware access belong at the production gateway/platform.

Never commit credentials, private support documents, raw production queries, or generated databases.

