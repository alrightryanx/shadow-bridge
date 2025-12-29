#!/usr/bin/env python
import sys
sys.path.insert(0, '.')
from web.app import create_app

app = create_app()
print("Registered routes:")
for rule in app.url_map.iter_rules():
    print(f"  {rule.rule} -> {rule.endpoint}")
