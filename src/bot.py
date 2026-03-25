"""
Playwright-based bot for scraping Sporty Hero crash game results.

Collects crash multiplier values from each round by observing the game.
The crash value is the multiplier at which the game "crashes" — if you
cash out before then, you win stake × multiplier.
"""

import argparse
import asyncio
import re
from pathlib import Path
from datetime import datetime, timezone

from playwright.async_api import async_playwright, Page

from src.db import get_connection, init_db, insert_crash, insert_crashes_bulk, get_total_stats

# Persistent browser profile directory (separate from soccer bot)
PROFILE_DIR = Path(__file__).parent.parent / "data" / "browser_profile"

# SportyBet Sporty Hero URL — update after inspection
SPORTY_HERO_URL = "https://www.sportybet.com/ng/games"


async def launch_browser(headless: bool = False, persistent: bool = True) -> tuple:
    """Launch browser with persistent profile."""
    pw = await async_playwright().start()

    if persistent:
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            context = await pw.chromium.launch_persistent_context(
                str(PROFILE_DIR),
                headless=headless,
                viewport={"width": 1280, "height": 900},
                args=["--disable-blink-features=AutomationControlled"],
            )
            page = context.pages[0] if context.pages else await context.new_page()
            return pw, context, page
        except Exception as e:
            print(f"⚠️  Persistent profile failed ({e})")

    browser = await pw.chromium.launch(
        headless=headless,
        args=["--disable-blink-features=AutomationControlled"],
    )
    context = await browser.new_context(viewport={"width": 1280, "height": 900})
    page = await context.new_page()
    return pw, context, page


async def wait_for_game(page: Page):
    """Navigate to Sporty Hero and wait for the game to load."""
    print("\n" + "=" * 60)
    print("SPORTY HERO — CRASH GAME DATA COLLECTOR")
    print("=" * 60)

    await page.goto(SPORTY_HERO_URL, wait_until="domcontentloaded", timeout=30000)
    print("\n1. Please log in if not already logged in.")
    print("2. Navigate to Sporty Hero game (under Exclusive).")
    print("\nPress Enter when the game is visible...")
    await asyncio.get_event_loop().run_in_executor(None, input)
    print("Ready to scrape!\n")
    return page


def _get_game_frame(page: Page):
    """
    Get the Sporty Hero game iframe.

    The game runs inside an iframe with id="games-lobby".
    Returns the Frame object, or None.
    """
    frame = page.frame(name="games-lobby")
    if frame:
        return frame
    # Fallback: try matching by URL fragment
    for f in page.frames:
        if "sportygames" in (f.url or ""):
            return f
    return None


async def scrape_crash_value(page: Page) -> dict | None:
    """
    Scrape the most recent crash value from the history row.

    The game iframe contains a coefficient-row with recent crash pills.
    Each pill has structure: div.coefficent-value > span with text like "1.20x".

    Returns:
        dict with 'crash_value' and 'round_id', or None
    """
    try:
        frame = _get_game_frame(page)
        if not frame:
            print("  ⚠️  Game iframe not found")
            return None

        # Get the first (most recent) crash value from the history row
        result = await frame.evaluate(r"""() => {
            const spans = document.querySelectorAll('div.coefficent-value span');
            if (spans.length === 0) return null;
            const txt = (spans[0].textContent || '').trim();
            const match = txt.match(/^(\d+\.?\d*)x$/i);
            if (!match) return null;
            return { value: parseFloat(match[1]), text: txt };
        }""")

        if result:
            return {
                "crash_value": result["value"],
                "round_id": f"R{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            }

        return None
    except Exception as e:
        print(f"❌ Error scraping crash value: {e}")
        return None


async def scrape_crash_history(page: Page) -> list[float]:
    """
    Scrape all crash values from the history row.

    The coefficient-row inside the game iframe shows the last N crash values.
    Selector: div.coefficent-value span (leaf spans with text like "1.20x").
    """
    try:
        frame = _get_game_frame(page)
        if not frame:
            return []

        result = await frame.evaluate(r"""() => {
            const values = [];
            // Scope to the coefficient-row to avoid duplicate elements
            const row = document.querySelector('div.coefficient-row');
            if (!row) return [];
            const chips = row.querySelectorAll('div.sh-multiplier-chip');
            for (const chip of chips) {
                const span = chip.querySelector('div.coefficent-value span');
                if (!span) continue;
                const txt = (span.textContent || '').trim();
                const match = txt.match(/^(\d+\.?\d*)x$/i);
                if (match) {
                    values.push(parseFloat(match[1]));
                }
            }
            return values;
        }""")
        return result or []
    except Exception:
        return []


async def run_scraper(rounds: int = 0, headless: bool = False) -> None:
    """Main scraping loop.

    Strategy: poll the crash history row every cycle. The history row shows
    the last N crash values, so we bulk-insert any we haven't seen before.
    This is more robust than trying to catch a single crash event.
    """
    conn = get_connection()
    init_db(conn)

    pw, context, page = await launch_browser(headless=headless)

    try:
        target = await wait_for_game(page)

        # Verify game iframe is accessible
        frame = _get_game_frame(target)
        if not frame:
            print("⚠️  Game iframe not found. Make sure you're on the Sporty Hero page.")
            print("    Looking for iframe with id='games-lobby'...")
            return

        print("✅ Game iframe found!\n")

        seen_values: list[float] = []  # Track the last history snapshot to detect new values
        total_inserted = 0
        poll_count = 0
        fail_count = 0  # Consecutive failed reads

        while rounds == 0 or total_inserted < rounds:
            poll_count += 1
            print(f"\n{'─' * 40}")
            print(f"Poll #{poll_count}")

            history = await scrape_crash_history(target)

            if history:
                fail_count = 0
                if not seen_values:
                    # First poll — just snapshot the history, don't insert
                    # (these are old rounds we can't timestamp accurately)
                    seen_values = history
                    print(f"  📋 Snapshot: {len(history)} existing values")
                    print(f"  ⏳ Waiting for new rounds...")
                else:
                    # Find new values by comparing with last snapshot
                    # New values appear at the start of the list
                    new_values = []
                    for val in history:
                        if val == seen_values[0]:
                            break
                        new_values.append(val)

                    if new_values:
                        now = datetime.now(timezone.utc)
                        crashes = [
                            {
                                "crash_value": v,
                                "round_id": f"R{now.strftime('%Y%m%d%H%M%S')}_{i}",
                                "timestamp": now.isoformat(),
                            }
                            for i, v in enumerate(new_values)
                        ]
                        inserted = insert_crashes_bulk(conn, crashes)
                        total_inserted += inserted
                        print(f"  💥 New crashes: {[f'{v:.2f}x' for v in new_values]}")
                        print(f"  📥 Inserted: {inserted} | Total: {total_inserted}")
                    else:
                        print(f"  ⏳ No new crashes yet (history: {len(history)} values)")

                    seen_values = history

                stats = get_total_stats(conn)
                print(f"  📊 DB: {stats['total_rounds']} rounds, "
                      f"avg: {stats['avg_crash']:.2f}x, "
                      f"<2x: {stats['below_2.0x']:.1f}%")
            else:
                fail_count += 1
                print(f"  ⚠️  Could not read crash history (fail #{fail_count})")
                if fail_count >= 5:
                    print("  🔄 Too many failures, refreshing page...")
                    try:
                        await target.reload(wait_until="domcontentloaded", timeout=30000)
                        await asyncio.sleep(5)
                        frame = _get_game_frame(target)
                        if frame:
                            print("  ✅ Page refreshed, iframe found")
                            fail_count = 0
                            seen_values = []  # Reset snapshot after refresh
                        else:
                            print("  ❌ Iframe not found after refresh")
                    except Exception as e:
                        print(f"  ❌ Refresh failed: {e}")

            # Wait before next poll
            if rounds == 0 or total_inserted < rounds:
                print("  Waiting 10s...")
                await asyncio.sleep(10)

    except KeyboardInterrupt:
        print("\n\n🛑 Stopped by user.")
    finally:
        stats = get_total_stats(conn)
        print(f"\n{'=' * 40}")
        print(f"SESSION SUMMARY")
        print(f"  Rounds: {stats['total_rounds']}")
        print(f"  Avg crash: {stats['avg_crash']:.2f}x")
        print(f"  Min: {stats['min_crash']:.2f}x | Max: {stats['max_crash']:.2f}x")
        print(f"  Below 2x: {stats['below_2.0x']:.1f}%")
        print(f"{'=' * 40}\n")

        conn.close()
        await context.close()
        await pw.stop()


async def _auto_inspect(wait_seconds: int = 45):
    """Non-interactive inspect: opens browser, waits, then auto-dumps everything."""
    pw, context, page = await launch_browser(headless=False, persistent=True)
    try:
        await page.goto(SPORTY_HERO_URL, wait_until="domcontentloaded", timeout=30000)
        print("\n" + "=" * 60)
        print("AUTO-INSPECT MODE")
        print("=" * 60)
        print("Browser is open. You have time to:")
        print("  1. Log in (if needed)")
        print("  2. Navigate to the Sporty Hero game")
        print(f"\nAuto-dump will start in {wait_seconds} seconds...\n")

        for remaining in range(wait_seconds, 0, -1):
            if remaining % 10 == 0 or remaining <= 5:
                print(f"  ⏱️  {remaining}s remaining...")
            await asyncio.sleep(1)

        print("\n🔍 Starting auto-dump...\n")

        data_dir = Path(__file__).parent.parent / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Screenshot
        await page.screenshot(path=str(data_dir / "hero_inspect.png"), full_page=True)
        print("📸 Screenshot saved to data/hero_inspect.png")

        # HTML dump
        html = await page.content()
        (data_dir / "hero_inspect.html").write_text(html, encoding="utf-8")
        print("📄 HTML saved to data/hero_inspect.html")

        # Scan for crash-like elements (Nx pattern)
        crash_els = await page.evaluate(r"""() => {
            const results = [];
            const allEls = document.querySelectorAll('*');
            for (const el of allEls) {
                if (el.children.length > 10) continue;
                const txt = (el.textContent || '').trim();
                if (txt.length > 0 && txt.length < 20) {
                    const match = txt.match(/\d+\.?\d*x/i);
                    if (match) {
                        const rect = el.getBoundingClientRect();
                        const style = getComputedStyle(el);
                        results.push({
                            tag: el.tagName.toLowerCase(),
                            cls: (el.className || '').substring(0, 80),
                            text: txt,
                            id: el.id || '',
                            fontSize: style.fontSize,
                            color: style.color,
                            w: Math.round(rect.width),
                            h: Math.round(rect.height),
                            x: Math.round(rect.x),
                            y: Math.round(rect.y),
                            parent: el.parentElement ? {
                                tag: el.parentElement.tagName.toLowerCase(),
                                cls: (el.parentElement.className || '').substring(0, 80),
                            } : null,
                            path: (() => {
                                const parts = [];
                                let cur = el;
                                while (cur && cur !== document.body) {
                                    let sel = cur.tagName.toLowerCase();
                                    if (cur.id) sel += '#' + cur.id;
                                    else if (cur.className) sel += '.' + cur.className.split(' ')[0];
                                    parts.unshift(sel);
                                    cur = cur.parentElement;
                                }
                                return parts.join(' > ');
                            })(),
                        });
                    }
                }
            }
            return results.slice(0, 50);
        }""")

        print(f"\n🎯 Elements matching crash pattern ({len(crash_els)}):\n")
        for i, e in enumerate(crash_els):
            print(f"  [{i:2d}] <{e['tag']}> \"{e['text']}\"")
            print(f"       class: {e['cls'] or '(none)'}")
            print(f"       size: {e['w']}x{e['h']}  pos: ({e['x']},{e['y']})  font: {e['fontSize']}  color: {e['color']}")
            print(f"       path: {e['path']}")
            if e['parent']:
                print(f"       parent: <{e['parent']['tag']}> cls={e['parent']['cls']}")
            print()

        # Also look for iframes (crash games often live in iframes)
        iframes = await page.evaluate("""() => {
            return Array.from(document.querySelectorAll('iframe')).map(f => ({
                src: f.src || '',
                id: f.id || '',
                cls: (f.className || '').substring(0, 80),
                w: f.width || f.offsetWidth,
                h: f.height || f.offsetHeight,
            }));
        }""")

        if iframes:
            print(f"\n🖼️  Iframes found ({len(iframes)}):")
            for i, f in enumerate(iframes):
                print(f"  [{i}] src={f['src'][:120]}  id={f['id']}  cls={f['cls']}  size={f['w']}x{f['h']}")

            # Try to scan inside iframes too
            for i, iframe_el in enumerate(await page.query_selector_all("iframe")):
                try:
                    frame = await iframe_el.content_frame()
                    if frame:
                        frame_crashes = await frame.evaluate(r"""() => {
                            const results = [];
                            const allEls = document.querySelectorAll('*');
                            for (const el of allEls) {
                                if (el.children.length > 10) continue;
                                const txt = (el.textContent || '').trim();
                                if (txt.length > 0 && txt.length < 20) {
                                    const match = txt.match(/\d+\.?\d*x/i);
                                    if (match) {
                                        const style = getComputedStyle(el);
                                        results.push({
                                            tag: el.tagName.toLowerCase(),
                                            cls: (el.className || '').substring(0, 80),
                                            text: txt,
                                            id: el.id || '',
                                            fontSize: style.fontSize,
                                            color: style.color,
                                            path: (() => {
                                                const parts = [];
                                                let cur = el;
                                                while (cur && cur !== document.body) {
                                                    let sel = cur.tagName.toLowerCase();
                                                    if (cur.id) sel += '#' + cur.id;
                                                    else if (cur.className) sel += '.' + cur.className.split(' ')[0];
                                                    parts.unshift(sel);
                                                    cur = cur.parentElement;
                                                }
                                                return parts.join(' > ');
                                            })(),
                                        });
                                    }
                                }
                            }
                            return results.slice(0, 30);
                        }""")

                        if frame_crashes:
                            print(f"\n  🎯 Inside iframe [{i}] — crash elements ({len(frame_crashes)}):")
                            for j, e in enumerate(frame_crashes):
                                print(f"    [{j:2d}] <{e['tag']}> \"{e['text']}\"  cls={e['cls']}  font={e['fontSize']}  color={e['color']}")
                                print(f"         path: {e['path']}")
                except Exception as ex:
                    print(f"  ⚠️  Could not access iframe [{i}]: {ex}")
        else:
            print("\n🖼️  No iframes found (game is in the main page)")

        # Canvas detection (some crash games use canvas)
        canvases = await page.evaluate("""() => {
            return Array.from(document.querySelectorAll('canvas')).map(c => ({
                id: c.id || '',
                cls: (c.className || '').substring(0, 80),
                w: c.width,
                h: c.height,
            }));
        }""")
        if canvases:
            print(f"\n🎨 Canvas elements ({len(canvases)}):")
            for c in canvases:
                print(f"  id={c['id']}  cls={c['cls']}  size={c['w']}x{c['h']}")

        print("\n✅ Inspection complete. Check data/hero_inspect.png and data/hero_inspect.html")

    finally:
        await context.close()
        await pw.stop()


async def _inspect_mode():
    """Launch browser to inspect Sporty Hero DOM (interactive)."""
    pw, context, page = await launch_browser(headless=False, persistent=False)
    try:
        await page.goto(SPORTY_HERO_URL, wait_until="domcontentloaded", timeout=30000)
        print("Browser launched. Navigate to Sporty Hero, then come back here.")
        input("Press Enter when the game is loaded...")

        # Take screenshot
        data_dir = Path(__file__).parent.parent / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        await page.screenshot(path=str(data_dir / "hero_inspect.png"), full_page=True)
        print(f"📸 Screenshot saved to data/hero_inspect.png")

        # Dump HTML
        html = await page.content()
        (data_dir / "hero_inspect.html").write_text(html, encoding="utf-8")
        print(f"📄 HTML saved to data/hero_inspect.html")

        # Look for crash-related elements
        crash_els = await page.evaluate(r"""() => {
            const results = [];
            const allEls = document.querySelectorAll('*');
            for (const el of allEls) {
                if (el.children.length > 10) continue;
                const txt = (el.textContent || '').trim();
                if (txt.length > 0 && txt.length < 20) {
                    const match = txt.match(/\d+\.?\d*x/i);
                    if (match) {
                        results.push({
                            tag: el.tagName.toLowerCase(),
                            cls: (el.className || '').substring(0, 60),
                            text: txt,
                            id: el.id || '',
                        });
                    }
                }
            }
            return results.slice(0, 30);
        }""")

        print(f"\n🎯 Crash-like elements ({len(crash_els)}):")
        for e in crash_els:
            print(f"  [{e['tag']}] {e['text']}  id={e['id']}  cls={e['cls']}")

        # Interactive mode
        print("\nInteractive inspect mode:")
        print("  screenshot, dump, select <css>, selectall <css>, eval <js>, quit")
        while True:
            cmd = input("inspect> ").strip()
            if cmd == "quit":
                break
            elif cmd == "screenshot":
                ts = datetime.now().strftime("%H%M%S")
                await page.screenshot(path=str(data_dir / f"hero_{ts}.png"), full_page=True)
                print("  Saved.")
            elif cmd == "dump":
                ts = datetime.now().strftime("%H%M%S")
                (data_dir / f"hero_{ts}.html").write_text(await page.content(), encoding="utf-8")
                print("  Saved.")
            elif cmd.startswith("select "):
                sel = cmd[7:].strip()
                el = await page.query_selector(sel)
                if el:
                    print(f"  {(await el.text_content() or '')[:200]}")
                else:
                    print("  Not found.")
            elif cmd.startswith("selectall "):
                sel = cmd[10:].strip()
                els = await page.query_selector_all(sel)
                for i, el in enumerate(els[:20]):
                    print(f"  [{i}] {(await el.text_content() or '')[:100]}")
            elif cmd.startswith("eval "):
                try:
                    result = await page.evaluate(cmd[5:].strip())
                    print(f"  {result}")
                except Exception as e:
                    print(f"  Error: {e}")

    finally:
        await context.close()
        await pw.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Sporty Hero Crash Game — Data Collector"
    )
    parser.add_argument(
        "--rounds", type=int, default=0,
        help="Number of rounds to scrape (0 = unlimited)"
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run in headless mode"
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="Launch interactive DOM inspector"
    )
    parser.add_argument(
        "--auto-inspect", action="store_true",
        help="Non-interactive inspect: opens browser, waits, auto-dumps"
    )
    parser.add_argument(
        "--wait", type=int, default=45,
        help="Seconds to wait before auto-dump (default: 45)"
    )
    args = parser.parse_args()

    if args.auto_inspect:
        asyncio.run(_auto_inspect(wait_seconds=args.wait))
    elif args.inspect:
        asyncio.run(_inspect_mode())
    else:
        asyncio.run(run_scraper(rounds=args.rounds, headless=args.headless))


if __name__ == "__main__":
    main()
