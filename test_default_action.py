#!/usr/bin/env python
"""Test that action_type defaults to skip."""
from server.app import app
from fastapi.testclient import TestClient

client = TestClient(app)

# Reset first
print("Resetting environment...")
r = client.post("/reset")
print(f"  Status: {r.status_code}")

# Test step with empty action (should default to "skip")
print("Testing POST /step with empty action_type (should default to 'skip')")
r = client.post("/step", json={})
print(f"  Status: {r.status_code}")
if r.status_code == 200:
    response = r.json()
    print(f"  ✓ Success! Last action: {response.get('last_action')}")
else:
    print(f"  Error: {r.json()}")
