from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ask_professor_g.cli import main


if __name__ == "__main__":
    main(["--cached", "--example", "examples/cached", "--config", "examples/demo_config.yaml", *sys.argv[1:]])
