"""Poll ngrok's local API (port 4040) for the tunnel URL and write it to /tmp/phoenix-swe-url."""
import sys
import time
import urllib.request
import json

for attempt in range(20):
    try:
        with urllib.request.urlopen("http://localhost:4040/api/tunnels", timeout=2) as resp:
            data = json.loads(resp.read())
        tunnels = data.get("tunnels", [])
        for t in tunnels:
            url = t.get("public_url", "")
            if url.startswith("https"):
                with open("/tmp/phoenix-swe-url", "w") as f:
                    f.write(url)
                print(f"  Ngrok tunnel: {url}")
                print(f"  Webhook URL:  {url}/webhook")
                sys.exit(0)
    except Exception:
        pass
    time.sleep(1)

print("  WARNING: could not reach ngrok API — check /tmp/ngrok-swe.log")
sys.exit(1)
