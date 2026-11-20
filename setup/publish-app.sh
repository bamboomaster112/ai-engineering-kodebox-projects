#!/usr/bin/env bash
# Publish AI Desk at http://<server-ip>/app/ through the KodeBox nginx, behind basic auth.
# Creates a random password once and keeps it in ~/aidesk-credentials.txt (readable only by you).
# Needs sudo for the two nginx files. Run again any time; it keeps the existing password.
set -euo pipefail
cd "$(dirname "$0")/.."

CRED="$HOME/aidesk-credentials.txt"
if [ ! -s "$CRED" ]; then
  umask 077
  PASS="$(openssl rand -base64 18 | tr -d '/+=' | cut -c1-20)"
  printf 'AI Desk web login\nuser: desk\npassword: %s\n' "$PASS" > "$CRED"
fi
PASS="$(sed -n 's/^password: //p' "$CRED")"
HASH="$(printf '%s' "$PASS" | openssl passwd -apr1 -stdin)"

printf 'desk:%s\n' "$HASH" | sudo tee /etc/nginx/aidesk.htpasswd >/dev/null
sudo chown root:www-data /etc/nginx/aidesk.htpasswd
sudo chmod 640 /etc/nginx/aidesk.htpasswd
sudo install -m 644 setup/nginx-aidesk.conf /etc/nginx/kodebox.d/aidesk.conf
sudo nginx -t
sudo systemctl reload nginx
echo "Published at http://<this server's IP>/app/  (login details in $CRED)"
