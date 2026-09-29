"""Run the LPR system as a service: camera workers, the REST API and the web app.

    python scripts/serve.py --config configs/system.yaml
    python scripts/serve.py --config configs/system.yaml --no-api   # cameras only

Then open the printed address in a browser, either on this computer or on a
phone connected to the same Wi-Fi. On first start the web app asks you to
create the admin account.

Events are stored in the SQLite database set in the config, and alerts go
to the log and the optional webhook. Stop with Ctrl+C.
"""
import argparse
import logging
import signal
import socket
import threading

import _path  # noqa: F401
from lpr.config import load_config
from lpr.stream import LPRSystem

log = logging.getLogger("lpr.serve")


def lan_addresses() -> list[str]:
    """This machine's LAN IPs, so a phone on the same Wi-Fi can connect."""
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))  # no packet is sent. This only picks the route
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/system.yaml")
    ap.add_argument("--no-api", action="store_true")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--ssl-certfile", default=None,
                    help="serve HTTPS (needed to install the web app on a phone's home screen)")
    ap.add_argument("--ssl-keyfile", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    system = LPRSystem(cfg)
    # Load the models now so problems show up at startup. On failure the
    # service still runs and the web app shows the error.
    if system.load_models():
        log.info("models loaded")

    if args.no_api:
        if system.model_error:
            raise SystemExit(f"cannot run cameras: {system.model_error}")
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

    host = args.host or cfg["api"]["host"]
    port = args.port or cfg["api"]["port"]
    scheme = "https" if args.ssl_certfile else "http"
    urls = [f"{scheme}://localhost:{port}"]
    if host in ("0.0.0.0", "::"):
        urls += [f"{scheme}://{ip}:{port}" for ip in lan_addresses()]
    print("\n" + "=" * 64)
    print("  PlateVision LPR is starting. Open the web app at:")
    for u in urls:
        print(f"    {u}")
    print("  (on a phone, use the address with your computer's IP; same Wi-Fi)")
    if system.model_error:
        print(f"\n  WARNING: models not loaded - {system.model_error}")
    print("=" * 64 + "\n")
    uvicorn.run(create_app(system), host=host, port=port, log_level=args.log_level.lower(),
                ssl_certfile=args.ssl_certfile, ssl_keyfile=args.ssl_keyfile)


if __name__ == "__main__":
    main()
