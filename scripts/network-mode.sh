#!/bin/sh

# Recover or inspect macOS networking when Clash Verge and EasyConnect coexist.
# This script never modifies the routing table, so EasyConnect-owned routes stay
# intact. It only manages the Wi-Fi system proxy left by Clash Verge.

set -eu

SERVICE="Wi-Fi"
CLASH_HOST="127.0.0.1"
CLASH_PORT="7897"

usage() {
    echo "Usage: $0 check | direct | clash"
    echo "  check  Show proxy state and representative public/private routes"
    echo "  direct Disable all Wi-Fi proxies; keep EasyConnect routes untouched"
    echo "  clash  Configure the Clash system proxy and private-network bypasses"
}

check_network() {
    echo "== HTTP proxy =="
    networksetup -getwebproxy "$SERVICE"
    echo "== HTTPS proxy =="
    networksetup -getsecurewebproxy "$SERVICE"
    echo "== SOCKS proxy =="
    networksetup -getsocksfirewallproxy "$SERVICE"
    echo "== EasyConnect/private route =="
    route -n get 172.16.30.53 | sed -n '/gateway:/p;/interface:/p'
    echo "== Public route =="
    route -n get 8.8.8.8 | sed -n '/gateway:/p;/interface:/p'
}

case "${1:-}" in
    check)
        check_network
        ;;
    direct)
        networksetup -setwebproxystate "$SERVICE" off
        networksetup -setsecurewebproxystate "$SERVICE" off
        networksetup -setsocksfirewallproxystate "$SERVICE" off
        echo "System proxy disabled. EasyConnect routes were not changed."
        check_network
        ;;
    clash)
        networksetup -setwebproxy "$SERVICE" "$CLASH_HOST" "$CLASH_PORT"
        networksetup -setsecurewebproxy "$SERVICE" "$CLASH_HOST" "$CLASH_PORT"
        networksetup -setsocksfirewallproxy "$SERVICE" "$CLASH_HOST" "$CLASH_PORT"
        networksetup -setproxybypassdomains "$SERVICE" \
            127.0.0.1 localhost '*.local' '<local>' \
            10.0.0.0/8 172.16.0.0/12 192.168.0.0/16
        echo "Clash system proxy enabled with private-network bypasses."
        check_network
        ;;
    *)
        usage
        exit 2
        ;;
esac
