"""Capture full WebSocket session + post-game seeds; analyze for early leaks.

For each round: log every WS frame, then fetch /round/{id}/seeds to get the
ground truth, then check whether any early frame leaked it.
"""
import websocket, requests, time, uuid, json, threading, re

OUT = open("/Users/eremie/Clones/probodds-parlay/tmp/sh_ws_capture.jsonl", "w")
def log(d):
    d["t"] = time.time()
    OUT.write(json.dumps(d) + "\n")
    OUT.flush()

sess = requests.Session()
sess.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "text/html"})
sess.get("https://www.sportybet.com/ng/m/games/sporty-hero", timeout=10)

ALL_TOPICS = [
    "/topic/ng-multiplier-wap",
    "/topic/ng-round-wap",
    "/topic/ng-lastRoundMultiplier-wap",
    # speculative topics - try anyway
    "/topic/ng-roundBet-wap",
    "/topic/ng-bet-wap",
    "/topic/ng-cashout-wap",
    "/topic/ng-seed-wap",
    "/topic/ng-roomInfo-wap",
    "/topic/ng-info-wap",
]

cookie_str = "; ".join(f"{c.name}={c.value}" for c in sess.cookies)
device_id = str(uuid.uuid4())
ws = websocket.create_connection(
    "wss://www.sportybet.com/ws/ng/games/sporty-hero/v1/game",
    header=[f"Cookie: {cookie_str}", "Origin: https://www.sportybet.com", "User-Agent: Mozilla/5.0"],
    subprotocols=["v12.stomp", "v11.stomp", "v10.stomp"],
    timeout=15,
)
ws.send(f"CONNECT\naccept-version:1.0,1.1,1.2\nheart-beat:10000,10000\ncountry-code:ng\nx-device-id:{device_id}\nuser-agent:Mozilla/5.0\ncontent-type:application/json\n\n\x00")
print("CONNECT sent, awaiting CONNECTED…")
msg = ws.recv()
print("Got:", repr(msg)[:200])
log({"ev": "connect", "msg": str(msg)[:500]})

for t in ALL_TOPICS:
    ws.send(f"SUBSCRIBE\nid:s-{t}\ndestination:{t}\n\n\x00")
    print("subscribed", t)

# Periodically fetch round to know current id
def fetch_round_loop():
    while True:
        try:
            r = sess.get("https://www.sportybet.com/api/ng/games/sporty-hero/v1/round", timeout=5)
            log({"ev": "round_poll", "data": r.json().get("data")})
        except Exception as e:
            log({"ev": "round_poll_err", "err": str(e)})
        time.sleep(3)
threading.Thread(target=fetch_round_loop, daemon=True).start()

# Main recv loop
DURATION = 300  # 5 minutes
start = time.time()
ws.settimeout(20)
while time.time() - start < DURATION:
    try:
        msg = ws.recv()
        if isinstance(msg, bytes):
            msg = msg.decode("utf-8", errors="replace")
        # Parse STOMP frame
        # MESSAGE\nheaders\n\nbody\x00
        m = re.match(r"^([A-Z]+)\n([^\x00]*?)\n\n([^\x00]*)\x00", msg, re.S)
        if m:
            cmd, hdr_block, body = m.groups()
            headers = dict(line.split(":", 1) for line in hdr_block.split("\n") if ":" in line)
            sub = headers.get("subscription", "")
            try:
                body_j = json.loads(body)
            except:
                body_j = body
            log({"ev": "stomp", "cmd": cmd, "sub": sub, "body": body_j})
            # Print interesting events
            if isinstance(body_j, dict):
                mt = body_j.get("messageType", "?")
                rid = body_j.get("roundId", "?")
                cm = body_j.get("currentMultiplier", "")
                # check for any seed-like field
                for key in ("serverSeed", "clientSeed", "seed", "houseCoefficient", "hash", "decimal"):
                    if key in body_j:
                        print(f"!!! LEAK: {key}={body_j[key]} in {sub} mt={mt}")
                if mt in ("ROUND_WAITING", "ROUND_PRE_START", "ROUND_END_WAIT", "ROUND_NEXT", "ROUND_END") or sub.endswith("lastRoundMultiplier-wap") or sub.endswith("round-wap"):
                    print(f"  {sub.split('-')[-1] if '-' in sub else sub} | round={rid} mt={mt} mult={cm} body={body_j}")
        else:
            log({"ev": "raw", "msg": msg[:500]})
    except websocket.WebSocketTimeoutException:
        print("(idle)")
    except Exception as e:
        print("recv err:", e)
        break

ws.close()
OUT.close()
print("Done. Captured to sh_ws_capture.jsonl")
