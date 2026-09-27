"""Run the LPR system as a service: camera workers plus the REST API.

    python scripts/serve.py --config configs/system.yaml
    python scripts/serve.py --config configs/system.yaml --no-api   # cameras only

Events are stored in the SQLite database set in the config, and alerts go
to the log and the optional webhook. Stop with Ctrl+C.
"""
import argparse
import logging
import signal
import threading

import _path  # noqa: F401
from lpr.config import load_config
from lpr.stream import LPRSystem


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/system.yaml")
    ap.add_argument("--no-api", action="store_true")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    system = LPRSystem(cfg)
    # Load the models now so configuration errors show up before serving.
    system.image_pipeline

    if args.no_api:
        stop = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        system.start()
        while not stop.is_set() and any(w.is_alive() for w in system.cameras.values()):
            stop.wait(1.0)
        system.stop()
        return

    import uvicorn

    from service.api import create_app

    uvicorn.run(create_app(system), host=args.host or cfg["api"]["host"],
                port=args.port or cfg["api"]["port"], log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
