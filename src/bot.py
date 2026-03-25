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

from src.db import get_connection, init_db, insert_crash, get_total_stats

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


async def scrape_crash_value(page) -> dict | None:
    """
    Scrape the crash value from the current/completed round.

    This needs to be adapted to the actual Sporty Hero DOM.
    Run --inspect mode first to discover the correct selectors.

    Returns:
        dict with 'crash_value' and 'round_id', or None
    """
    # ── PLACEHOLDER: Update after DOM inspection ─────────
    # The crash value is typically displayed as a large number like "2.47x"
    # after the round ends. The round history might show recent values.
    #
    # Use --inspect mode to find the correct selectors.

    try:
        # Try to find crash value from the game display
        result = await page.evaluate(r"""() => {
            // Search for crash value displays
            // Common patterns: "2.47x", "Crashed at 2.47x", etc.
            const allEls = document.querySelectorAll('*');
            const crashes = [];
            for (const el of allEls) {
                if (el.children.length > 5) continue;
                const txt = (el.textContent || '').trim();
                // Match patterns like "2.47x" or "2.47X"
                const match = txt.match(/^(\d+\.?\d*)x$/i);
                if (match) {
                    crashes.push({
                        value: parseFloat(match[1]),
                        text: txt,
                        tag: el.tagName.toLowerCase(),
                        cls: (el.className || '').substring(0, 50),
                    });
                }
            }
            return crashes;
        }""")

        if result:
            # Take the most prominent crash value
            return {
                "crash_value": result[0]["value"],
                "round_id": f"R{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            }

        return None
    except Exception as e:
        print(f"❌ Error scraping crash value: {e}")
        return None


async def scrape_crash_history(page) -> list[float]:
    """
    Scrape the crash history displayed on screen.

    Many crash games show the last N crash values in a row.
    This lets us collect multiple data points at once.
    """
    try:
        result = await page.evaluate(r"""() => {
            const values = [];
            const allEls = document.querySelectorAll('*');
            for (const el of allEls) {
                if (el.children.length > 0) continue;
                const txt = (el.textContent || '').trim();
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
    """Main scraping loop."""
    conn = get_connection()
    init_db(conn)

    pw, context, page = await launch_browser(headless=headless)

    try:
        target = await wait_for_game(page)

        rounds_scraped = 0
        while rounds == 0 or rounds_scraped < rounds:
            print(f"\n{'─' * 40}")
            print(f"Round #{rounds_scraped + 1}")

            # Scrape crash value
            crash = await scrape_crash_value(target)

            if crash:
                inserted = insert_crash(conn, crash["round_id"], crash["crash_value"])
                if inserted:
                    print(f"  💥 Crash: {crash['crash_value']:.2f}x")
                    stats = get_total_stats(conn)
                    print(f"  📊 Total: {stats['total_rounds']} rounds, "
                          f"avg: {stats['avg_crash']:.2f}x, "
                          f"<2x: {stats['below_2.0x']:.1f}%")
                else:
                    print(f"  Already scraped, skipping.")
            else:
                print("  ⚠️  Could not scrape crash value")

            rounds_scraped += 1

            # Wait for next round
            if rounds == 0 or rounds_scraped < rounds:
                print("  Waiting for next round...")
                await asyncio.sleep(10)  # Adjust based on round duration

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


async def _inspect_mode():
    """Launch browser to inspect Sporty Hero DOM."""
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
        help="Launch DOM inspector"
    )
    args = parser.parse_args()

    if args.inspect:
        asyncio.run(_inspect_mode())
    else:
        asyncio.run(run_scraper(rounds=args.rounds, headless=args.headless))


if __name__ == "__main__":
    main()
