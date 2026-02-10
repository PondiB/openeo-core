"""Process registry – discover and query openEO process JSON specs."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any


class ProcessRegistry:
    """Registry of openEO process specification JSON files.

    The specs are loaded from package data so they work from installed
    wheels.

    Usage::

        registry = ProcessRegistry()
        procs = registry.list_processes()
        ndvi_spec = registry.get_process("ndvi")
    """

    def __init__(self) -> None:
        self._specs: dict[str, dict[str, Any]] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_processes(self) -> list[str]:
        """Return sorted list of available process IDs."""
        return sorted(self._specs.keys())

    def get_process(self, process_id: str) -> dict[str, Any]:
        """Return the full JSON spec for a process.

        Raises
        ------
        KeyError
            If the process ID is not found.
        """
        try:
            return self._specs[process_id]
        except KeyError:
            raise KeyError(
                f"Process {process_id!r} not found. "
                f"Available: {self.list_processes()}"
            ) from None

    def search(self, text: str) -> list[dict[str, Any]]:
        """Search processes whose id or description contains *text* (case-insensitive)."""
        text_lower = text.lower()
        results: list[dict[str, Any]] = []
        for spec in self._specs.values():
            haystack = (
                spec.get("id", "") + " " + spec.get("description", "")
            ).lower()
            if text_lower in haystack:
                results.append(spec)
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load all JSON files from the packaged resources directory."""
        # Use importlib.resources for compatibility with wheels
        pkg = "openeo_core.processes.resources.process_specs.current"
        try:
            # Python 3.9+
            spec_dir = resources.files(pkg)  # type: ignore[attr-defined]
            for entry in spec_dir.iterdir():
                if entry.name.endswith(".json"):
                    data = json.loads(entry.read_text(encoding="utf-8"))
                    pid = data.get("id", entry.name.removesuffix(".json"))
                    self._specs[pid] = data
        except (TypeError, AttributeError, ModuleNotFoundError):
            # Fallback – resolve via __file__
            here = Path(__file__).resolve().parent
            spec_path = here / "resources" / "process_specs" / "current"
            if spec_path.is_dir():
                for fp in sorted(spec_path.glob("*.json")):
                    data = json.loads(fp.read_text(encoding="utf-8"))
                    pid = data.get("id", fp.stem)
                    self._specs[pid] = data
