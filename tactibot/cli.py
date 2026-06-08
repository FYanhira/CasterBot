"""CLI CASTERBOT."""

from __future__ import annotations

import argparse
import sys

from tactibot.config.loader import load_config
from tactibot.pipeline import run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CASTERBOT — pipeline SAM 3 para fútbol robótico (FutBotMX)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Ejecutar pipeline completo Fase 1")
    run_p.add_argument(
        "--config",
        "-c",
        default="configs/example_short.yaml",
        help="Ruta al YAML de configuración",
    )

    args = parser.parse_args(argv)
    if args.command == "run":
        cfg = load_config(args.config)
        out_dir = run_pipeline(cfg)
        print(f"Pipeline completado. Salidas en: {out_dir}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
