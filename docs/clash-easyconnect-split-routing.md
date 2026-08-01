# Clash Verge + EasyConnect split routing

## Intended traffic flow

- Public internet: Wi-Fi `en0`, with applications using the Clash system proxy.
- Corporate/private networks: bypass Clash and follow the routes installed by
  EasyConnect on its `utun` interface.

## Clash rule enhancement

The following rules must appear before subscription proxy rules:

```yaml
prepend:
  - IP-CIDR,172.16.0.0/12,DIRECT,no-resolve
  - IP-CIDR,10.0.0.0/8,DIRECT,no-resolve
  - IP-CIDR,192.168.0.0/16,DIRECT,no-resolve
  - DOMAIN-SUFFIX,local,DIRECT
  - DOMAIN-SUFFIX,lan,DIRECT
```

`DIRECT` means that Clash does not select a proxy node. macOS then applies its
normal routing table, where EasyConnect owns the corporate routes.

## Clash settings

- System Proxy: enabled.
- TUN Mode: disabled unless an application cannot honor the system proxy.
- System proxy bypass: `127.0.0.1`, `localhost`, `*.local`, `10.0.0.0/8`,
  `172.16.0.0/12`, `192.168.0.0/16`.
- Do not configure a Clash default route over an EasyConnect `utun` interface.

If TUN mode is enabled later, also add the private ranges to Mihomo
`route-exclude-address`; otherwise Clash and EasyConnect may compete for them.

## Open the configuration folders

Open the Clash Verge configuration folder in Finder:

```bash
open "$HOME/Library/Application Support/io.github.clash-verge-rev.clash-verge-rev"
```

Open the profile/rule enhancement folder directly:

```bash
open "$HOME/Library/Application Support/io.github.clash-verge-rev.clash-verge-rev/profiles"
```

Open this repository's network helper folder:

```bash
open "/Users/jxl/Desktop/visual_rl/scripts"
```

Open the split-routing guide itself with the default application:

```bash
open "/Users/jxl/Desktop/visual_rl/docs/clash-easyconnect-split-routing.md"
```

## Turning Clash off safely

With TUN disabled, turning Clash off should restore ordinary public access over
Wi-Fi while EasyConnect continues to own its private routes. If Clash exits
abnormally, macOS may retain a system proxy pointing at the now-closed local
port `127.0.0.1:7897`. That makes public applications appear offline even
though the routing table is healthy.

Use the repository helper to clear only the system proxy:

```bash
sudo "/Users/jxl/Desktop/visual_rl/scripts/network-mode.sh" direct
```

This does not delete or replace any EasyConnect route. To restore the Clash
system proxy and private-network bypass list:

```bash
sudo "/Users/jxl/Desktop/visual_rl/scripts/network-mode.sh" clash
```

To inspect the state without changing anything:

```bash
"/Users/jxl/Desktop/visual_rl/scripts/network-mode.sh" check
```

Alternatively, open `/Users/jxl/Desktop/visual_rl` in Finder and double-click
`network-mode.command`. It presents a menu and uses `sudo` only for operations
that change the system proxy.

## Verification

Run after connecting EasyConnect and starting Clash:

```bash
route -n get 172.16.30.53
route -n get 8.8.8.8
scutil --proxy
```

Expected results:

- `172.16.30.53` uses the EasyConnect `utun` interface.
- `8.8.8.8` uses `en0`.
- the proxy exception list contains the three private IPv4 ranges.

For SSH diagnosis, use:

```bash
ping -c 10 172.16.30.53
ssh -o ConnectTimeout=10 -o ServerAliveInterval=10 -p 22 root@172.16.30.53
```

If the route is correct but packet loss remains high, the problem is the
EasyConnect/VPN link rather than Clash routing.
