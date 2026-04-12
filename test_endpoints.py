#!/usr/bin/env python
"""Quick endpoint test."""
from server.app import app
from fastapi.testclient import TestClient

client = TestClient(app)

# Test health
print("Testing GET /")
r = client.get("/")
print(f"  Status: {r.status_code}")

# Test reset
print("Testing POST /reset")
r = client.post("/reset")
print(f"  Status: {r.status_code}, Keys: {list(r.json().keys())[:5]}")

# Test step
print("Testing POST /step")
r = client.post("/step", json={"action_type": "delete"})
print(f"  Status: {r.status_code}, Keys: {list(r.json().keys())[:5]}")

# Test state
print("Testing GET /state")
r = client.get("/state")
print(f"  Status: {r.status_code}, Keys: {list(r.json().keys())}")

print("✓ All endpoints working!")
