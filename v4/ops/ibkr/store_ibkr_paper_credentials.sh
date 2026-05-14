#!/usr/bin/env bash
set -euo pipefail

USERNAME_SERVICE="${IBKR_USERNAME_SERVICE:-autoresearch-trading-ibkr-paper-username}"
PASSWORD_SERVICE="${IBKR_PASSWORD_SERVICE:-autoresearch-trading-ibkr-paper-password}"

printf "IBKR paper username: "
IFS= read -r IBKR_USERNAME
if [[ -z "$IBKR_USERNAME" ]]; then
  echo "Username cannot be empty." >&2
  exit 2
fi

printf "IBKR paper password: "
IFS= read -rs IBKR_PASSWORD
printf "\n"
if [[ -z "$IBKR_PASSWORD" ]]; then
  echo "Password cannot be empty." >&2
  exit 2
fi

security add-generic-password -a "$USER" -s "$USERNAME_SERVICE" -w "$IBKR_USERNAME" -U >/dev/null
security add-generic-password -a "$USER" -s "$PASSWORD_SERVICE" -w "$IBKR_PASSWORD" -U >/dev/null

echo "Stored IBKR paper credentials in macOS Keychain services:"
echo "  $USERNAME_SERVICE"
echo "  $PASSWORD_SERVICE"
