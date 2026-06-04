"""Rendering POC: feed randeval bit streams into a Monte Carlo renderer.

Two adapters live here:
- mitsuba_adapter: registers a Python Sampler that pulls floats from a numpy bit array
- integrator: pure-Python fallback if Mitsuba's Python Sampler trampoline does not work
"""
from __future__ import annotations

from .bitstream import BitFloatStream

__all__ = ["BitFloatStream"]
