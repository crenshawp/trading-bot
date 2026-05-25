"""CLI entry point: python -m trading_bot <command>"""

import argparse
import getpass
import sys

from trading_bot import secrets


def cmd_secrets_list() -> None:
    set_secrets = secrets.list_secrets()
    for name in sorted(secrets.KNOWN_SECRETS):
        status = "SET" if name in set_secrets else "NOT SET"
        print(f"  {name:<30} {status}")


def cmd_secrets_set(name: str) -> None:
    value = getpass.getpass(f"Value for {name}: ")
    secrets.set_secret(name, value.strip())
    print(f"Secret '{name}' stored.")


def cmd_secrets_get(name: str) -> None:
    from trading_bot.config import RUNTIME
    if RUNTIME == "railway":
        print("Cannot read secrets to stdout on Railway.", file=sys.stderr)
        sys.exit(1)
    value = secrets.get_secret(name)
    if value is None:
        print(f"Secret '{name}' is not set.", file=sys.stderr)
        sys.exit(1)
    print(value)


def cmd_secrets_delete(name: str) -> None:
    secrets.delete_secret(name)
    print(f"Secret '{name}' deleted.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="trading_bot")
    sub = parser.add_subparsers(dest="command", required=True)

    secrets_parser = sub.add_parser("secrets")
    secrets_sub = secrets_parser.add_subparsers(dest="secrets_cmd", required=True)

    secrets_sub.add_parser("list")

    set_p = secrets_sub.add_parser("set")
    set_p.add_argument("name")

    get_p = secrets_sub.add_parser("get")
    get_p.add_argument("name")

    del_p = secrets_sub.add_parser("delete")
    del_p.add_argument("name")

    args = parser.parse_args()

    if args.command == "secrets":
        if args.secrets_cmd == "list":
            cmd_secrets_list()
        elif args.secrets_cmd == "set":
            cmd_secrets_set(args.name)
        elif args.secrets_cmd == "get":
            cmd_secrets_get(args.name)
        elif args.secrets_cmd == "delete":
            cmd_secrets_delete(args.name)


if __name__ == "__main__":
    main()
