import websocket, requests, time, uuid

sess = requests.Session()
sess.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126",
    "Accept": "text/html,application/xhtml+xml",
})
r = sess.get("https://www.sportybet.com/ng/m/games/sporty-hero", allow_redirects=True, timeout=10)
print("home", r.status_code, "cookies:", list(sess.cookies.keys()))

for url in [
    "https://www.sportybet.com/api/ng/onboarding-svc/lobby/games/sporty-hero",
    "https://www.sportybet.com/api/ng/auth/refresh-tokens",
    "https://www.sportybet.com/api/ng/games/sporty-hero/v1/round",
]:
    try:
        r = sess.get(url, timeout=10)
        print(url, r.status_code, list(sess.cookies.keys()))
    except Exception as e:
        print(url, "EXC", e)

print("=== final cookies ===")
for c in sess.cookies:
    print(f"  {c.name}={c.value[:40]}...")

cookie_str = "; ".join(f"{c.name}={c.value}" for c in sess.cookies)
device_id = str(uuid.uuid4())
headers = [
    f"Cookie: {cookie_str}",
    "Origin: https://www.sportybet.com",
    "User-Agent: Mozilla/5.0",
]
print("=== opening WS with cookies ===")
ws = websocket.create_connection(
    "wss://www.sportybet.com/ws/ng/games/sporty-hero/v1/game",
    header=headers,
    subprotocols=["v12.stomp", "v11.stomp", "v10.stomp"],
    timeout=10,
)
connect = (
    "CONNECT\naccept-version:1.0,1.1,1.2\nheart-beat:10000,10000\n"
    "country-code:ng\n"
    f"x-device-id:{device_id}\n"
    "user-agent:Mozilla/5.0\ncontent-type:application/json\n\n\x00"
)
ws.send(connect)
print(">> CONNECT sent")
ws.settimeout(8)
try:
    while True:
        msg = ws.recv()
        print("<<", repr(msg)[:800])
        if isinstance(msg, str) and msg.startswith("CONNECTED"):
            for topic in ["/topic/ng-multiplier-wap", "/topic/ng-round-wap", "/topic/ng-lastRoundMultiplier-wap"]:
                ws.send(f"SUBSCRIBE\nid:s-{topic}\ndestination:{topic}\n\n\x00")
                print(">> SUB", topic)
        if isinstance(msg, str) and msg.startswith("ERROR"):
            break
except Exception as e:
    print("recv exc:", e)
ws.close()
