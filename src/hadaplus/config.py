from pathlib import Path
import yaml

def load_config(path: str | Path) -> dict:
    """Load a YAML configuration file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
