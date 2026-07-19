import yaml
from pathlib import Path
from typing import Dict, Any

import logging

logger = logging.getLogger(__name__)

def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
