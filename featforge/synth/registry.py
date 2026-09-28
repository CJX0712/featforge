"""Synthesizer registry — name -> implementation, resolved dynamically."""
from __future__ import annotations

from typing import Dict

from ..core.errors import SynthError
from .synthesizer import FeatForgeSynth

SYNTHESIZERS = {
    FeatForgeSynth.name: FeatForgeSynth,
}


def get_synthesizer(name: str, **kw):
    if name not in SYNTHESIZERS:
        raise SynthError(f"unknown synthesizer: {name}")
    return SYNTHESIZERS[name](**kw)


def list_synthesizers() -> Dict[str, str]:
    return {k: v.__doc__.splitlines()[0] if v.__doc__ else "" for k, v in SYNTHESIZERS.items()}
