# Security policy

Retention Radar is a personal, open-source teaching project. It trains on synthetic data and serves
from a committed model bundle. It is not a production service.

## Supported versions

Only the `main` branch gets fixes. There are no releases with long-term support.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Report it privately through
[GitHub private vulnerability reporting](https://github.com/santoshshinde2012/retention-radar/security/advisories/new)
and include the steps to reproduce, the affected file or endpoint, and the impact you see.
If that form is not available, open an issue that only asks for a private contact. Leave the details out.

You can expect an acknowledgement within a week. The fix and the advisory are published together.

## Scope and known trade-offs

These are deliberate and documented, so they are not vulnerabilities on their own:

- **No authentication.** The FastAPI service (`make api`) and the Streamlit app (`make ui`) have no login.
  `make api` binds to 127.0.0.1. Put your own auth in front before exposing either one.
- **Synthetic data.** The records and subscribers are generated. Do not load real customer data without
  your own privacy review.
- **Model files.** `models/` holds joblib files, which run code when loaded. Load only the bundle from this repo
  or one you trained yourself.

Real issues include a secret committed to the repo, a way to make the API read or write files outside the repo,
or a pinned dependency with a known critical CVE.
