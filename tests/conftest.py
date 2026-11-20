import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_JAR = Path.home() / "Desktop" / "logisim.jar"
if "LOGISIM_JAR" not in os.environ and DEFAULT_JAR.exists():
    os.environ["LOGISIM_JAR"] = str(DEFAULT_JAR)

from logisim import jar  # noqa: E402

HAVE_JAR = jar.available()


def pytest_collection_modifyitems(config, items):
    if HAVE_JAR:
        return
    skip = pytest.mark.skip(reason="Logisim JAR / Java 21 not available")
    for item in items:
        if "jar" in item.keywords:
            item.add_marker(skip)
