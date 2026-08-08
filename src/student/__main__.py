from fire import Fire

from .cli import CLI


def main() -> None:
    """Run the CLI application using Fire."""
    Fire(CLI)


if __name__ == "__main__":
    main()
