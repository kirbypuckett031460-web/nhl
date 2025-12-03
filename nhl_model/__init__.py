"""NHL model package utilities.

This file exists to ensure the `nhl_model` directory is treated as a
standard Python package when running scripts directly (e.g., from a Windows
`C:\\nhl` folder).  Without it, `nhl_model` could be misidentified as a
non-package module on some environments, triggering import errors such as
`ModuleNotFoundError: No module named 'nhl_model.common'`.
"""

__all__ = ["common", "data_fetcher", "social"]
