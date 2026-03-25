"""Allow running as `python -m src bot`, `python -m src analyze`, etc."""
import sys

if len(sys.argv) < 2:
    print("Usage: python -m src <command> [options]")
    print("Commands: bot, analyze, strategies")
    sys.exit(1)

command = sys.argv.pop(1)

if command == "bot":
    from src.bot import main
    main()
elif command == "analyze":
    from src.analyze import main
    main()
elif command == "strategies":
    from src.strategies import main
    main()
else:
    print(f"Unknown command: {command}")
    sys.exit(1)
