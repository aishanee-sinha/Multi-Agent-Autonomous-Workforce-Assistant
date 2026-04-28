from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import requests
import json
import os
from dotenv import load_dotenv
load_dotenv()

CLIENT_ID     = os.getenv('GOOGLE_CALENDAR_CLIENT_ID')       # paste your client ID here (ends with .apps.googleusercontent.com)
CLIENT_SECRET = os.getenv('GOOGLE_CALENDAR_CLIENT_SECRET')      # paste your new secret here
REDIRECT_URI  = 'http://localhost:3000/callback'

print(f'✅ Loaded CLIENT_ID: {CLIENT_ID[:20]}...')

class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if 'code' in params:
            code = params['code'][0]
            print(f'\n✅ Got auth code!')

            response = requests.post('https://oauth2.googleapis.com/token', data={
                'code': code,
                'client_id': CLIENT_ID,
                'client_secret': CLIENT_SECRET,
                'redirect_uri': REDIRECT_URI,
                'grant_type': 'authorization_code',
            })

            tokens = response.json()
            print('\n🎉 Tokens:')
            print(json.dumps(tokens, indent=2))

            if 'refresh_token' in tokens:
                name = input('\nEnter friend name to save as: ')
                with open(f'tokens_{name}.json', 'w') as f:
                    json.dump(tokens, f, indent=2)
                print(f'💾 Saved to tokens_{name}.json')
            else:
                print('⚠️ No refresh_token received! See note below.')

            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'<h2>Done! You can close this tab.</h2>')

        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'No code found.')

    def log_message(self, format, *args):
        pass  # Suppress server logs

print('🚀 Waiting on http://localhost:3000/callback ...')
print('\n👉 Send this auth URL to your friend:')
print(f'''
https://accounts.google.com/o/oauth2/v2/auth?client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}&response_type=code&scope=https://www.googleapis.com/auth/calendar.readonly&access_type=offline&prompt=consent
''')

HTTPServer(('localhost', 3000), CallbackHandler).serve_forever()