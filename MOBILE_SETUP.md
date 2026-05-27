# LankaMind — Mobile & Wi-Fi Network Setup

Connect every device on your Wi-Fi to the LankaMind AI network — **no app install,
no configuration, no permissions required** on client devices.

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    Your Wi-Fi Router                            │
│                                                                 │
│  PC / Laptop (server)          Phone / Tablet (client)         │
│  ┌───────────────────┐         ┌──────────────────────┐        │
│  │  lankamind serve  │ ◄──────►│  Browser (Chrome/    │        │
│  │                   │  HTTP   │  Safari / Firefox)   │        │
│  │  Gateway :5700    │         │                      │        │
│  │  Workers :5500-02 │         │  Opens:              │        │
│  │  API     :8000    │         │  lankamind.local:8000│        │
│  │                   │         │  or 192.168.x.x:8000 │        │
│  │  mDNS broadcasts: │         │                      │        │
│  │  lankamind.local  │         └──────────────────────┘        │
│  └───────────────────┘                                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 1 — Server Setup (the PC that runs the AI)

### Step 1: Install LankaMind

```bash
# Windows / macOS / Linux
git clone https://github.com/shalinda-j/lankamind.git
cd lankamind
pip install -e .
```

### Step 2: Start Everything with One Command

```bash
lankamind serve
```

Or with Python directly:
```bash
python scripts/launch_all.py
```

**What this starts automatically:**
- Gateway (orchestrates workers)
- 3 Worker shards (holds the AI model split across processes)
- REST API server on port 8000
- mDNS announcement (so phones find it by name)

**First run:** GPT-2 model (~500 MB) downloads automatically. Wait ~2 minutes.

### Step 3: Read the Banner

After startup you will see:

```
┌─────────────────────────────────────────────────────────┐
│  ✅  LankaMind is READY                                  │
├─────────────────────────────────────────────────────────┤
│  Open on this machine : http://localhost:8000           │
│  Open on your phone   : http://192.168.1.5:8000         │
│  mDNS (auto-discover) : http://lankamind.local:8000     │
├─────────────────────────────────────────────────────────┤
│  API docs             : http://localhost:8000/docs      │
│  Wi-Fi QR code        : http://localhost:8000/v1/...    │
└─────────────────────────────────────────────────────────┘
```

---

## Part 2 — Connecting a Phone (Client — No Install Required)

### Android (Chrome / Samsung Browser / Firefox)

1. Make sure the phone is on the **same Wi-Fi** as the PC
2. Open your browser and type the URL from the banner:
   - `http://lankamind.local:8000`  ← works automatically on most networks
   - OR `http://192.168.1.5:8000`   ← use the IP shown in the banner
3. The LankaMind web app loads instantly — no install, no permissions
4. Type a prompt, tap **⚡ Generate**

### iPhone / iPad (Safari)

1. Same Wi-Fi as the PC
2. Open Safari and go to `http://lankamind.local:8000`
   - If `.local` doesn't resolve: use the IP address from the banner
3. Tap **Share → Add to Home Screen** to install it like an app (PWA)

### Windows / macOS laptop on the same Wi-Fi

```
http://lankamind.local:8000
```
or the IP address from the banner.

---

## Part 3 — Android as a Worker Node (Termux)

An Android phone can **contribute compute** to the network, earning LKM tokens.

### Step 1: Install Termux

Download from [F-Droid](https://f-droid.org/packages/com.termux/) (free, no Google Play needed).

### Step 2: Install Python in Termux

```bash
# In Termux terminal:
pkg update && pkg upgrade -y
pkg install python python-pip -y
```

### Step 3: Install LankaMind

```bash
pip install lankamind
```

Or from source:
```bash
pkg install git
git clone https://github.com/shalinda-j/lankamind.git
cd lankamind
pip install -e .
```

### Step 4: Find Your Phone's IP Address

In Termux:
```bash
ip addr show wlan0 | grep 'inet '
# Output: inet 192.168.1.8/24 ...
# Your phone's IP is: 192.168.1.8
```

### Step 5: Join the Network as a Worker

```bash
lankamind node \
  --gateway tcp://192.168.1.5:5700 \
  --host 192.168.1.8
```

Replace:
- `192.168.1.5` = the PC's IP (shown in the banner)
- `192.168.1.8` = your phone's IP

The phone will now download its model shard (~500 MB) and start contributing compute.

### Step 6: Check Your Balance

```bash
lankamind balance
```

---

## Part 4 — Multiple PCs / Laptops as Workers

Any machine on the Wi-Fi can be a worker:

**Machine 2 (worker):**
```bash
lankamind node \
  --gateway tcp://192.168.1.5:5700 \
  --host 192.168.1.6 \
  --shards 3
```

**Machine 3 (another worker):**
```bash
lankamind node \
  --gateway tcp://192.168.1.5:5700 \
  --host 192.168.1.7
```

The gateway automatically builds the inference pipeline from all registered workers.

---

## Troubleshooting

### "lankamind.local doesn't load on my phone"
- Some Windows networks block mDNS. Use the IP address instead.
- On Windows: install [Bonjour Print Services](https://support.apple.com/kb/DL999) to enable `.local` resolution.
- Try `http://192.168.1.X:8000` directly (check the IP in the banner).

### "I can't connect to the PC from my phone"
- Make sure both devices are on the **same Wi-Fi** (not one on 2.4GHz, one on 5GHz with AP isolation)
- Windows Firewall may block port 8000. Allow it:
  ```
  netsh advfirewall firewall add rule name="LankaMind" dir=in action=allow protocol=TCP localport=8000
  ```
- Or temporarily disable Windows Firewall for Private networks.

### "Timeout after 60 seconds"
- Workers are not ready yet. Wait for the banner to appear.
- Check `logs/worker_0.log` for errors.

### "No workers connected" shown in the web UI
- Start workers with `lankamind serve` (not just `lankamind api`)
- Or run `lankamind node` in a separate terminal first

### Termux: "pip install lankamind fails"
```bash
pkg install build-essential python-numpy  # some deps need build tools
pip install --no-build-isolation lankamind
```

### Want GPU acceleration?

Install CUDA workers on a desktop with a GPU:
```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cu121
lankamind node --gateway tcp://SERVER:5700 --host MY-IP
```

---

## Quick Reference

| Command | What it does |
|---------|-------------|
| `lankamind serve` | Start everything (1-command setup) |
| `lankamind node --gateway tcp://IP:5700 --host MY-IP` | Join as a worker |
| `lankamind status --gateway tcp://IP:5701` | See all connected workers |
| `lankamind balance` | Check LKM token earnings |
| `lankamind keys` | Show this device's public key |

---

## Network Architecture

```
Wi-Fi Network (192.168.1.0/24)
│
├── PC / Server (192.168.1.5)
│   ├── Gateway         :5700 (worker heartbeats)
│   ├── Gateway         :5701 (client discovery)
│   ├── Worker 0        :5500 (model layers 0-3)
│   ├── Worker 1        :5501 (model layers 4-7)
│   ├── Worker 2        :5502 (model layers 8-11)
│   ├── REST API        :8000 (web UI + /v1/complete)
│   └── mDNS            :5353 (announces lankamind.local)
│
├── Android Phone (192.168.1.8) — CLIENT
│   └── Browser → http://192.168.1.5:8000
│
├── iPhone (192.168.1.9) — CLIENT
│   └── Safari → http://lankamind.local:8000
│
└── Laptop 2 (192.168.1.6) — WORKER (optional)
    └── lankamind node --gateway tcp://192.168.1.5:5700
```

---

*For more help: [GitHub Issues](https://github.com/shalinda-j/lankamind/issues)*
