"""SLM rank-stability study.

Does a small-language-model leaderboard measured at full precision still hold
after 4-bit on-device quantization? This package builds quantized variants,
scores every variant with one fixed harness, and compares the resulting
leaderboards with rank statistics.
"""

__version__ = "0.1.0"
