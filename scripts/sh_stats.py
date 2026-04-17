import json, statistics
from collections import Counter
seeds, mults, csets = [], [], []
with open("sh_rounds.jsonl") as f:
    for line in f:
        d = json.loads(line)
        ss = d.get("serverSeeds", [])
        if ss: seeds.append(ss[0])
        if "houseCoefficient" in d: mults.append(float(d["houseCoefficient"]))
        csets.append([c.get("clientSeed") for c in d.get("clientSeeds", [])])
print(f"Total rounds: {len(seeds)}")
print(f"Seed lengths: {set(len(s) for s in seeds)}")
print(f"First 10 seeds: {seeds[:10]}")
print(f"Last 10 seeds:  {seeds[-10:]}")
ints = [int(s, 16) for s in seeds]
print(f"\n48-bit space coverage:")
print(f"  Min: {min(ints):>15}  ({min(ints)/(2**48)*100:.2f}% of 2^48)")
print(f"  Max: {max(ints):>15}  ({max(ints)/(2**48)*100:.2f}% of 2^48)")
print(f"  Mean: {statistics.mean(ints):>14.0f}  Expected: {(2**48)/2:.0f}")
print(f"  StdDev: {statistics.stdev(ints):>12.0f}  Expected: {(2**48)/(12**0.5):.0f}")

all_chars = "".join(seeds)
c = Counter(all_chars)
print(f"\nHex char frequencies (45*12 = {len(all_chars)} chars):")
exp = len(all_chars) / 16
for ch in "0123456789abcdef":
    cnt = c.get(ch, 0)
    bar = "*" * cnt
    print(f"  {ch}: {cnt:3d}  (exp {exp:.1f})  {bar}")

diffs = [(ints[i+1] - ints[i]) % (2**48) for i in range(len(ints)-1)]
print(f"\nSequential diffs (mod 2^48): mean = {statistics.mean(diffs)/2**48*100:.2f}% of range (expected ~50)")

print(f"\n=== Multiplier fairness ({len(mults)} rounds) ===")
print(f"Mean multiplier: {statistics.mean(mults):.2f}x")
for X in (1.5, 2, 5, 10, 20, 50, 100):
    obs = sum(1 for m in mults if m >= X) / len(mults)
    expected = 0.97 / X
    print(f"  P(>={X:5.1f}x): obs={obs*100:5.1f}%  expected={expected*100:5.1f}%  delta={(obs-expected)*100:+5.1f}pp")

sizes = [len(c) for c in csets]
print(f"\nBettors per round: min={min(sizes)}, mean={statistics.mean(sizes):.1f}, max={max(sizes)}")
