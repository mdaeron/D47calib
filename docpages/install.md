# 1. Installation

This should do the trick:

```bash
pip install D47calib
```

Alternatively:

1. download or clone the source from [https://github.com/mdaeron/D47calib]()
2. chose one of one of the following options:
	+ copy/move the `/src/D47calib` directory to somewhere in your Python path
	+ copy/move the `/src/D47calib` directory to your current working directory
	+ copy/move the `/src/D47calib` directory to any other location (e.g., `/foo/bar`) and include the following code snippet in your scripts:

```py
import sys
sys.path.append('/foo/bar')
```

I you don't install from pip, you will probably need to install the requirements listed in `pyproject.toml`.
