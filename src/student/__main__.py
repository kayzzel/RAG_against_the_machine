from fire import Fire

from .cli import CLI


def main() -> None:
    Fire(CLI)


if __name__ == "__main__":
    main()
