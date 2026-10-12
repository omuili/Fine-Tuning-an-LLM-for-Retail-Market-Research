"""SBA 928: fine-tuning flan-t5-small for retail market research."""


def main() -> None:
    from .pipeline import run

    run()
