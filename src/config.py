from pathlib import Path
import yaml

def load_config(config_path: str | Path = "config.yaml") -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)