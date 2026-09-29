"""Manage web-app user accounts from the command line.

    python scripts/manage_users.py add admin --role admin        # prompts for the password
    python scripts/manage_users.py add gate1 --role operator
    python scripts/manage_users.py list
    python scripts/manage_users.py remove gate1
    python scripts/manage_users.py passwd admin                  # reset a password

Roles: viewer (dashboard, search, export), operator (+ scan, watchlist,
cameras), admin (+ users, audit log).
"""
import argparse
import getpass

import _path  # noqa: F401
from lpr.auth import ROLES, AuthManager
from lpr.config import load_config
from lpr.storage import EventStore


def ask_password() -> str:
    while True:
        p1 = getpass.getpass("Password (min. 8 characters): ")
        if p1 == getpass.getpass("Repeat password: "):
            return p1
        print("Passwords do not match, try again.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/system.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("username")
    a.add_argument("--role", default="operator", choices=ROLES)
    a.add_argument("--password", default=None, help="omit to be prompted (safer)")
    sub.add_parser("list")
    r = sub.add_parser("remove")
    r.add_argument("username")
    pw = sub.add_parser("passwd")
    pw.add_argument("username")
    args = ap.parse_args()

    cfg = load_config(args.config)
    s = cfg["storage"]
    store = EventStore(s["database"], s["snapshot_dir"])
    auth = AuthManager(store)

    if args.cmd == "list":
        users = store.list_users()
        if not users:
            print("no users (the web app will ask for an admin account on first visit)")
        for u in users:
            print(f"{u['username']:<24} {u['role']:<10} created {u['created_ts']}")
    elif args.cmd == "add":
        if store.get_user(args.username):
            raise SystemExit(f"user {args.username} already exists (use passwd to reset)")
        auth.create_user(args.username, args.password or ask_password(), args.role)
        store.audit("cli", "user_add", f"{args.username} ({args.role})")
        print(f"added {args.username} ({args.role})")
    elif args.cmd == "remove":
        try:
            deleted = auth.delete_user(args.username)
        except ValueError as e:
            raise SystemExit(str(e))
        if not deleted:
            raise SystemExit("unknown user")
        store.audit("cli", "user_delete", args.username)
        print(f"removed {args.username}")
    elif args.cmd == "passwd":
        user = store.get_user(args.username)
        if not user:
            raise SystemExit("unknown user")
        auth.create_user(args.username, ask_password(), user["role"])
        store.audit("cli", "user_passwd", args.username)
        print(f"password changed for {args.username}")


if __name__ == "__main__":
    main()
