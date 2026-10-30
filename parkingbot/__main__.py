"""Entry point: python -m parkingbot"""

from __future__ import annotations

import logging
import sys

from .config import ConfigError, load_settings


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Quiet the very chatty httpx request logger.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    # Imported here so config errors surface before pulling in heavy deps.
    from .bot import build_application

    app = build_application(settings)
    from .metrics import start_server

    start_server(app.bot_data["parking_bot"], settings)
    logging.getLogger(__name__).info("Starting bot (long polling)…")
    # drop_pending_updates avoids replaying a backlog after downtime.
    app.run_polling(drop_pending_updates=True, allowed_updates=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
