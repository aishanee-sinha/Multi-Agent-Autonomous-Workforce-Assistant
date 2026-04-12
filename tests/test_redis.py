import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv; load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
from redis_store import save_session, load_session, _get_client

sid = save_session({'test': 'hello'})
print('session_id:', sid)

# Peek — confirm the key exists in Redis before consuming it
raw = _get_client().get(sid)
print('raw value in Redis:', raw)

print('loaded:', load_session(sid))
print('second get (should be None):', load_session(sid))

