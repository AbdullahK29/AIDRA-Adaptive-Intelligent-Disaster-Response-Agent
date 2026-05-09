"""
main.py
───────
Entry point for AIDRA — Adaptive Intelligent Disaster Response Agent.

Run with:
    python main.py

Requirements:
    pip install numpy scikit-learn

All other imports are from the standard library (tkinter, heapq, math, etc.)
"""

import tkinter as tk
from gui import AIDRAGui


if __name__ == "__main__":
    root = tk.Tk()
    app  = AIDRAGui(root)
    root.mainloop()
