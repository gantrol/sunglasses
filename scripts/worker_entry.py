"""Frozen executable entry: CLI operations stay out of the native UI process."""
from sunglasses.worker import main

if __name__ == "__main__":
    raise SystemExit(main())
