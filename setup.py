from pathlib import Path
from setuptools import setup

CWD = Path(__file__).absolute().parent


def get_version():
    """Gets the project version"""
    path = CWD / "bayesian_lora" / "__init__.py"
    content = path.read_text()
    for line in content.splitlines():
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


if __name__ == "__main__":
    print(f"Version: {get_version()}")
    setup(name="bayesian_lora", version=get_version())
