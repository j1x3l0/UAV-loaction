#!/bin/sh

# Double-clickable macOS launcher for scripts/network-mode.sh.

set -u

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
NETWORK_SCRIPT="$SCRIPT_DIR/scripts/network-mode.sh"

if [ ! -x "$NETWORK_SCRIPT" ]; then
    echo "Cannot execute: $NETWORK_SCRIPT"
    echo "Run: chmod +x \"$NETWORK_SCRIPT\""
    printf "Press Enter to close..."
    read -r _
    exit 1
fi

echo "Clash Verge + EasyConnect network mode"
echo ""
echo "1) Check status (no changes)"
echo "2) Direct mode (disable Clash system proxy)"
echo "3) Clash mode (enable proxy with private-network bypasses)"
echo "q) Quit"
echo ""
printf "Select [1/2/3/q]: "
read -r choice

case "$choice" in
    1)
        "$NETWORK_SCRIPT" check
        ;;
    2)
        echo "macOS may request your administrator password."
        sudo "$NETWORK_SCRIPT" direct
        ;;
    3)
        echo "Start Clash Verge first. macOS may request your administrator password."
        sudo "$NETWORK_SCRIPT" clash
        ;;
    q|Q)
        exit 0
        ;;
    *)
        echo "Invalid selection: $choice"
        ;;
esac

echo ""
printf "Press Enter to close..."
read -r _
