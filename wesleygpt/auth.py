# Wesley wrote this
"""Bearer API keys. Keys come from WESLEYGPT_API_KEYS (comma-separated), which on
Cloud Run is mounted from Secret Manager -- never baked into the image."""
import hmac


def parse_api_keys(raw):
    keys = {k.strip() for k in (raw or "").split(",") if k.strip()}
    if not keys:
        raise ValueError("WESLEYGPT_API_KEYS is empty; set at least one API key (comma-separated)")
    return keys


def bearer_key_is_valid(header, keys):
    if not header:
        return False
    scheme, _, key = header.partition(" ")
    if scheme.lower() != "bearer" or not key.strip():
        return False
    presented = key.strip().encode()
    # Compare against every key so timing doesn't reveal which one nearly matched.
    return any([hmac.compare_digest(presented, k.encode()) for k in keys])
