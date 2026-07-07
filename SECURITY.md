# Security Policy

## Supported Version

The `main` branch is the supported branch.

## Reporting

Please open a private security advisory on GitHub for vulnerabilities. Do not
publish API keys, local `.env` files, or private deployment URLs in issues.

## Notes

The default Flask server binds to `127.0.0.1`. If you deploy it publicly, set
`WC26_TOKEN` so mutating POST endpoints require bearer auth.
