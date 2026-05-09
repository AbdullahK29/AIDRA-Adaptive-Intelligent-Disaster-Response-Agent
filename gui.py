import tkinter as tk
from tkinter import ttk, scrolledtext
import threading, re
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from environment import (
    GRID_SIZE, CELL_COLORS, SEVERITY_COLORS,
    BASE_POS, MEDICAL_CENTERS, INITIAL_GRID, INITIAL_VICTIMS,
)
from agent import AIDRAAgent

CELL_PX = 52
F_TITLE = ("Courier New", 16, "bold")
F_LABEL = ("Courier New", 11)
F_BOLD  = ("Courier New", 11, "bold")
F_SMALL = ("Courier New", 10)
F_LOG   = ("Courier New", 10)
F_KPIV  = ("Courier New", 11, "bold")


class AIDRAGui:
    def __init__(self, root):
        self.root         = root
        self.root.title("AIDRA — Adaptive Intelligent Disaster Response Agent")
        self.root.configure(bg="#0d1117")
        self.root.resizable(True, True)
        self.agent        = None
        self.path_results = []
        self.running      = False
        self.compare_rows = []
        self._build_ui()
        self._reset_agent()

    # ─────────────────────────────────────────────────────────
    #  LAYOUT  (3 columns)
    #
    #   col_left   │  col_mid        │  col_right
    #   ─────────────────────────────────────────
    #   Grid map   │  Decision Log   │  Controls
    #   Legend     │  (full height)  │  KPIs
    #              │                 │  Comparison + Graph
    #              │                 │  ML Metrics
    # ─────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── top bar ──────────────────────────────────────────
        top = tk.Frame(self.root, bg="#0d1117", pady=5)
        top.pack(fill=tk.X, padx=10)
        tk.Label(top, text="⚡ AIDRA", font=F_TITLE,
                 fg="#00e5ff", bg="#0d1117").pack(side=tk.LEFT)
        tk.Label(top, text="  Adaptive Intelligent Disaster Response Agent",
                 font=F_SMALL, fg="#546e7a", bg="#0d1117").pack(side=tk.LEFT)

        # ── 3-column body ────────────────────────────────────
        body = tk.Frame(self.root, bg="#0d1117")
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # LEFT: grid + legend
        col_left = tk.Frame(body, bg="#0d1117")
        col_left.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 6))

        sz = GRID_SIZE * CELL_PX + 2
        self.canvas = tk.Canvas(col_left, width=sz, height=sz, bg="#0d1117",
                                bd=0, highlightthickness=1,
                                highlightbackground="#00e5ff")
        self.canvas.pack(padx=4, pady=4)
        self._build_legend(col_left)

        # MIDDLE: decision log (full height)
        col_mid = tk.Frame(body, bg="#0d1117", width=310)
        col_mid.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 6))
        col_mid.pack_propagate(False)
        self._build_log(col_mid)

        # RIGHT: controls + KPIs + comparison + ML
        col_right = tk.Frame(body, bg="#0d1117", width=430)
        col_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        col_right.pack_propagate(False)
        self._build_controls(col_right)
        self._build_kpi_panel(col_right)
        self._build_compare_panel(col_right)
        self._build_ml_panel(col_right)

    # ── legend ────────────────────────────────────────────────
    def _build_legend(self, parent):
        leg = tk.Frame(parent, bg="#0d1117")
        leg.pack(fill=tk.X, padx=4, pady=(0, 2))
        items = [("Normal","#2d4a2d"),("Blocked","#3a0a0a"),
                 ("High Risk","#7a2200"),("Medical","#003d99"),("Base","#5a4400"),
                 ("Critical","#ff2222"),("Moderate","#ff9900"),
                 ("Minor","#44ff44"),("Path","#00e5ff")]
        for label, color in items:
            f = tk.Frame(leg, bg="#0d1117")
            f.pack(side=tk.LEFT, padx=2)
            tk.Label(f, bg=color, width=2, height=1).pack(side=tk.LEFT)
            tk.Label(f, text=label, font=("Courier New", 8),
                     fg="#90a4ae", bg="#0d1117").pack(side=tk.LEFT)

    # ── decision log (middle column, full height) ─────────────
    def _build_log(self, parent):
        f = tk.LabelFrame(parent, text=" Decision Log ", font=F_BOLD,
                          fg="#ffd740", bg="#131d2e", bd=1, relief=tk.SOLID)
        f.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

        self.log_box = scrolledtext.ScrolledText(
            f, font=F_LOG, bg="#0d1117", fg="#cfd8dc",
            bd=0, state=tk.DISABLED, wrap=tk.WORD)
        self.log_box.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        for tag, color in [
            ("INFO","#90a4ae"),("SYS","#546e7a"),("ML","#69ff47"),
            ("CSP","#ffd740"),("PLAN","#80deea"),("FUZZY","#ea80fc"),
            ("SEARCH","#4fc3f7"),("RESCUE","#00e5ff"),("KPI","#ff9800"),
            ("EVENT","#ff5252"),("REPLAN","#ce93d8"),("ERR","#ff1744")]:
            self.log_box.tag_config(tag, foreground=color)

    # ── controls ──────────────────────────────────────────────
    def _build_controls(self, parent):
        ctrl = tk.LabelFrame(parent, text=" Controls ", font=F_BOLD,
                             fg="#00e5ff", bg="#131d2e", bd=1, relief=tk.SOLID)
        ctrl.pack(fill=tk.X, pady=(0, 5))

        r1 = tk.Frame(ctrl, bg="#131d2e")
        r1.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(r1, text="Algorithm:", font=F_LABEL,
                 fg="#90a4ae", bg="#131d2e").pack(side=tk.LEFT)
        self.algo_var = tk.StringVar(value="A*")
        ttk.Combobox(r1, textvariable=self.algo_var, width=16, state="readonly",
                     font=F_LABEL,
                     values=["A*","BFS","DFS","Greedy",
                             "Hill Climbing","Sim. Annealing"]
                     ).pack(side=tk.LEFT, padx=(8,12))
        self.ml_status_var = tk.StringVar(value="● NOT TRAINED")
        self.ml_badge = tk.Label(r1, textvariable=self.ml_status_var,
                                 font=("Courier New",10,"bold"),
                                 fg="#ff5252", bg="#131d2e")
        self.ml_badge.pack(side=tk.LEFT)

        r2 = tk.Frame(ctrl, bg="#131d2e")
        r2.pack(fill=tk.X, padx=10, pady=3)
        tk.Label(r2, text="Risk Mode: ", font=F_LABEL,
                 fg="#90a4ae", bg="#131d2e").pack(side=tk.LEFT)
        self.risk_var = tk.StringVar(value="balanced")
        for val, lbl in [("fast","Fast"),("balanced","Balanced"),("safe","Safe")]:
            tk.Radiobutton(r2, text=lbl, variable=self.risk_var, value=val,
                           font=F_LABEL, fg="#cfd8dc", bg="#131d2e",
                           selectcolor="#1e3a5f", activebackground="#131d2e"
                           ).pack(side=tk.LEFT, padx=5)

        r3 = tk.Frame(ctrl, bg="#131d2e")
        r3.pack(fill=tk.X, padx=10, pady=6)
        self._btn(r3,"▶ RUN",     self._run,         "#00e5ff","#001a33").pack(side=tk.LEFT,padx=3)
        self._btn(r3,"↺ RESET",   self._reset_agent, "#ff6b35","#1a0d00").pack(side=tk.LEFT,padx=3)
        self._btn(r3,"🤖 TRAIN ML",self._train_ml,   "#69ff47","#0a1a00").pack(side=tk.LEFT,padx=3)

        tk.Label(ctrl, text="ℹ  Switch algorithm freely — no reset needed.",
                 font=("Courier New",9), fg="#546e7a", bg="#131d2e"
                 ).pack(pady=(0,5))

    # ── KPI panel ─────────────────────────────────────────────
    def _build_kpi_panel(self, parent):
        f = tk.LabelFrame(parent, text=" Last Run — KPIs ", font=F_BOLD,
                          fg="#00e5ff", bg="#131d2e", bd=1, relief=tk.SOLID)
        f.pack(fill=tk.X, pady=(0,5))
        self.kpi_vars = {}
        for key, label in [
            ("victims_saved",      "Victims Saved"),
            ("avg_rescue_time",    "Avg Rescue Time"),
            ("total_risk_exposure","Risk Exposure"),
            ("resource_util",      "Resource Util"),
            ("csp_backtracks",     "CSP Backtracks")]:
            row = tk.Frame(f, bg="#131d2e")
            row.pack(fill=tk.X, padx=12, pady=1)
            tk.Label(row, text=f"{label}:", font=F_LABEL, fg="#546e7a",
                     bg="#131d2e", width=20, anchor="w").pack(side=tk.LEFT)
            self.kpi_vars[key] = tk.StringVar(value="—")
            tk.Label(row, textvariable=self.kpi_vars[key],
                     font=F_KPIV, fg="#00e5ff", bg="#131d2e").pack(side=tk.LEFT)

    # ── comparison table + graph ──────────────────────────────
    def _build_compare_panel(self, parent):
        f = tk.LabelFrame(parent, text=" Algorithm Comparison ", font=F_BOLD,
                          fg="#ffd740", bg="#131d2e", bd=1, relief=tk.SOLID)
        f.pack(fill=tk.X, pady=(0,5))

        # table header
        hdr = tk.Frame(f, bg="#1e2d3d")
        hdr.pack(fill=tk.X, padx=4, pady=(4,0))
        for txt, w in [("Algorithm",13),("Saved",6),
                       ("AvgTime",8),("Risk",6),("Nodes",7),("BT",4)]:
            tk.Label(hdr, text=txt, font=("Courier New",10,"bold"),
                     fg="#ffd740", bg="#1e2d3d", width=w, anchor="w"
                     ).pack(side=tk.LEFT, padx=2)

        self.cmp_inner = tk.Frame(f, bg="#131d2e")
        self.cmp_inner.pack(fill=tk.X, padx=4, pady=(0,2))

        # buttons row
        btn_row = tk.Frame(f, bg="#131d2e")
        btn_row.pack(fill=tk.X, padx=6, pady=(0,5))
        self._btn(btn_row, "📊 Plot Graph", self._plot_graph,
                  "#ea80fc","#1a0033").pack(side=tk.LEFT, padx=(0,6))
        tk.Button(btn_row, text="✕ Clear", command=self._clear_compare,
                  font=("Courier New",9), fg="#ff6b35",
                  bg="#131d2e", bd=0, cursor="hand2").pack(side=tk.LEFT)

    def _redraw_compare(self):
        for w in self.cmp_inner.winfo_children():
            w.destroy()
        bgs = ["#0d1a2d","#131d2e"]
        for i, rd in enumerate(self.compare_rows):
            bg  = bgs[i % 2]
            row = tk.Frame(self.cmp_inner, bg=bg)
            row.pack(fill=tk.X)
            for val, w in zip(
                [rd["algo"],rd["saved"],rd["avg_time"],rd["risk"],rd["nodes"],rd["bt"]],
                [13, 6, 8, 6, 7, 4]
            ):
                tk.Label(row, text=val, font=F_SMALL, fg="#cfd8dc",
                         bg=bg, width=w, anchor="w").pack(side=tk.LEFT, padx=2)

    def _clear_compare(self):
        self.compare_rows.clear()
        self._redraw_compare()

    def _add_compare_row(self, algo, kpis, avg_nodes):
        self.compare_rows.append({
            "algo":     algo,
            "saved":    f"{kpis['victims_saved']}/{kpis['total_victims']}",
            "avg_time": str(kpis["avg_rescue_time"]),
            "risk":     str(kpis["total_risk_exposure"]),
            "nodes":    f"{avg_nodes:.0f}",
            "bt":       str(kpis["csp_backtracks"]),
        })
        self._redraw_compare()

    # ── graph popup ───────────────────────────────────────────
    def _plot_graph(self):
        if not self.compare_rows:
            return

        algos     = [r["algo"]     for r in self.compare_rows]
        avg_times = [float(r["avg_time"]) for r in self.compare_rows]
        risks     = [float(r["risk"])     for r in self.compare_rows]
        nodes     = [float(r["nodes"])    for r in self.compare_rows]

        # ── popup window ─────────────────────────────────────
        win = tk.Toplevel(self.root)
        win.title("Algorithm Performance Comparison")
        win.configure(bg="#0d1117")
        win.geometry("820x540")

        fig, axes = plt.subplots(1, 3, figsize=(10, 4.2))
        fig.patch.set_facecolor("#0d1117")

        bar_color  = ["#00e5ff","#ffd740","#69ff47","#ff6b35","#ea80fc","#ff5252"]
        bar_colors = [bar_color[i % len(bar_color)] for i in range(len(algos))]

        datasets = [
            (axes[0], avg_times, "Avg Rescue Time (steps)", "Avg Time"),
            (axes[1], risks,     "Risk Exposure (hazard steps)", "Risk Exposure"),
            (axes[2], nodes,     "Avg Nodes Expanded", "Nodes Expanded"),
        ]

        for ax, values, title, ylabel in datasets:
            ax.set_facecolor("#131d2e")
            bars = ax.bar(algos, values, color=bar_colors,
                          edgecolor="#0d1117", linewidth=0.8)
            ax.set_title(title, color="#cfd8dc", fontsize=9, pad=6)
            ax.set_ylabel(ylabel, color="#546e7a", fontsize=8)
            ax.tick_params(axis="x", colors="#90a4ae", labelsize=7, rotation=20)
            ax.tick_params(axis="y", colors="#546e7a", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#1e2d3d")
            ax.yaxis.label.set_color("#546e7a")
            # value labels on bars
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + max(values)*0.01,
                        f"{val:.1f}", ha="center", va="bottom",
                        color="#ffffff", fontsize=8, fontweight="bold")

        fig.suptitle("AIDRA — Algorithm Performance Comparison",
                     color="#00e5ff", fontsize=11, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        canvas_fig = FigureCanvasTkAgg(fig, master=win)
        canvas_fig.draw()
        canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True,
                                        padx=8, pady=8)
        tk.Button(win, text="✕  Close", command=win.destroy,
                  font=F_BOLD, fg="#ff6b35", bg="#1a0d00",
                  bd=1, relief=tk.SOLID, cursor="hand2"
                  ).pack(pady=(0,8))

    # ── ML panel ──────────────────────────────────────────────
    def _build_ml_panel(self, parent):
        f = tk.LabelFrame(parent, text=" ML Metrics (kNN | Naive Bayes) ",
                          font=F_BOLD, fg="#69ff47",
                          bg="#131d2e", bd=1, relief=tk.SOLID)
        f.pack(fill=tk.X, pady=(0,5))
        self.ml_text = tk.Text(f, height=4, font=F_LOG,
                               bg="#0a1a0a", fg="#69ff47",
                               bd=0, state=tk.DISABLED)
        self.ml_text.pack(fill=tk.X, padx=6, pady=6)

    # ── button helper ─────────────────────────────────────────
    def _btn(self, parent, text, cmd, fg, bg):
        return tk.Button(parent, text=text, command=cmd, font=F_BOLD,
                         fg=fg, bg=bg, activeforeground=fg, activebackground=bg,
                         bd=1, relief=tk.SOLID, cursor="hand2", padx=6, pady=3)

    # ─────────────────────────────────────────────────────────
    #  GRID DRAWING
    # ─────────────────────────────────────────────────────────

    def _draw_grid(self, paths=None):
        self.canvas.delete("all")
        grid    = self.agent.grid    if self.agent else INITIAL_GRID
        victims = self.agent.victims if self.agent else INITIAL_VICTIMS

        path_cells = set()
        if paths:
            for pr in paths:
                for cell in pr.get("path",[]):
                    path_cells.add(cell)

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                x1,y1 = c*CELL_PX+1, r*CELL_PX+1
                x2,y2 = x1+CELL_PX-2, y1+CELL_PX-2
                color = ("#00344d" if (r,c) in path_cells
                         else CELL_COLORS.get(grid[r][c],"#2d4a2d"))
                self.canvas.create_rectangle(x1,y1,x2,y2,
                    fill=color,outline="#1e2d3d",width=1)

        for mr,mc in MEDICAL_CENTERS:
            cx,cy = mc*CELL_PX+CELL_PX//2, mr*CELL_PX+CELL_PX//2
            self.canvas.create_rectangle(
                mc*CELL_PX+4,mr*CELL_PX+4,
                mc*CELL_PX+CELL_PX-4,mr*CELL_PX+CELL_PX-4,
                fill="#003d99",outline="#4fc3f7",width=2)
            self.canvas.create_text(cx,cy,text="🏥",font=("Arial",14))

        br,bc = BASE_POS
        self.canvas.create_rectangle(
            bc*CELL_PX+4,br*CELL_PX+4,
            bc*CELL_PX+CELL_PX-4,br*CELL_PX+CELL_PX-4,
            fill="#5a4400",outline="#ffd740",width=2)
        self.canvas.create_text(bc*CELL_PX+CELL_PX//2,br*CELL_PX+CELL_PX//2,
                                text="🚑",font=("Arial",13))

        for v in victims:
            vr,vc   = v["pos"]
            color   = SEVERITY_COLORS.get(v["severity"],"#fff")
            rescued = v.get("rescued",False)
            self.canvas.create_oval(
                vc*CELL_PX+8,vr*CELL_PX+8,
                vc*CELL_PX+CELL_PX-8,vr*CELL_PX+CELL_PX-8,
                fill="#222" if rescued else "#111",
                outline="#666" if rescued else color,width=2)
            self.canvas.create_text(
                vc*CELL_PX+CELL_PX//2, vr*CELL_PX+CELL_PX//2,
                text="✓" if rescued else str(v["id"]),
                font=("Courier New",11,"bold"),
                fill="#888" if rescued else color)
            if v.get("survival_prob") is not None:
                self.canvas.create_text(
                    vc*CELL_PX+CELL_PX//2, vr*CELL_PX+CELL_PX-7,
                    text=f"{v['survival_prob']:.0%}",
                    font=("Courier New",7),fill="#80deea")

        if paths:
            for pr in paths:
                path = pr.get("path",[])
                for i in range(min(pr.get("_draw_to",len(path)),len(path))):
                    r2,c2 = path[i]
                    self.canvas.create_rectangle(
                        c2*CELL_PX+16,r2*CELL_PX+16,
                        c2*CELL_PX+CELL_PX-16,r2*CELL_PX+CELL_PX-16,
                        fill="#00e5ff",outline="",stipple="gray25")

    # ─────────────────────────────────────────────────────────
    #  ACTIONS
    # ─────────────────────────────────────────────────────────

    def _reset_agent(self):
        self.agent = AIDRAAgent(log_callback=self._log)
        self.path_results = []
        self._draw_grid()
        self._clear_log()
        self._log("[SYS] Full reset — ML cleared.", "SYS")
        for k in self.kpi_vars:
            self.kpi_vars[k].set("—")
        self._refresh_ml_panel()
        self._update_ml_badge()

    def _train_ml(self):
        self._log("[ML] Training kNN + Naive Bayes...", "ML")
        def _do():
            self.agent.ml.train()
            self.root.after(0, self._refresh_ml_panel)
            self.root.after(0, self._update_ml_badge)
        threading.Thread(target=_do, daemon=True).start()

    def _run(self):
        if self.running:
            return
        self.running = True
        algo = self.algo_var.get()
        risk = self.risk_var.get()

        ml_backup = self.agent.ml
        self.agent.soft_reset()
        self.agent.ml = ml_backup

        self._clear_log()
        self._draw_grid()
        self._log(f"[SYS] Running {algo} | {risk} mode", "SYS")

        def _do():
            results   = self.agent.simulate_rescue(strategy=algo, risk_preference=risk)
            self.path_results = results
            avg_nodes = self._extract_avg_nodes()
            self.root.after(0, self._refresh_kpis)
            self.root.after(0, lambda: self._add_compare_row(algo, self.agent.kpis, avg_nodes))
            self.root.after(0, lambda: self._animate(results))
            self.root.after(0, self._refresh_ml_panel)
            self.root.after(0, self._update_ml_badge)
            self.running = False

        threading.Thread(target=_do, daemon=True).start()

    def _extract_avg_nodes(self):
        totals = [int(m.group(1))
                  for line in self.agent.decision_log
                  for m in [re.search(r'nodes_expanded=(\d+)', line)] if m]
        return sum(totals)/len(totals) if totals else 0

    # ── animation ─────────────────────────────────────────────

    def _animate(self, results):
        if not results:
            self._draw_grid(); return
        states = [{"path": r["path"], "_draw_to": 0} for r in results]
        total  = max(len(r["path"]) for r in results)
        def step(s):
            for ds in states:
                ds["_draw_to"] = min(s, len(ds["path"]))
            self._draw_grid(paths=states)
            if s <= total:
                self.root.after(90, lambda: step(s+1))
        step(0)

    # ── refresh helpers ───────────────────────────────────────

    def _refresh_kpis(self):
        k = self.agent.kpis
        if not k: return
        self.kpi_vars["victims_saved"].set(f"{k['victims_saved']}/{k['total_victims']}")
        self.kpi_vars["avg_rescue_time"].set(f"{k['avg_rescue_time']} steps")
        self.kpi_vars["total_risk_exposure"].set(f"{k['total_risk_exposure']} steps")
        self.kpi_vars["resource_util"].set(str(k["resource_util"]))
        self.kpi_vars["csp_backtracks"].set(str(k["csp_backtracks"]))

    def _refresh_ml_panel(self):
        self.ml_text.config(state=tk.NORMAL)
        self.ml_text.delete("1.0", tk.END)
        if self.agent and self.agent.ml.trained:
            for line in self.agent.ml.metrics_summary():
                self.ml_text.insert(tk.END, line+"\n")
        else:
            self.ml_text.insert(tk.END,
                "  Auto-trains on first RUN, or press 🤖 TRAIN ML.\n")
        self.ml_text.config(state=tk.DISABLED)

    def _update_ml_badge(self):
        if self.agent and self.agent.ml.trained:
            self.ml_status_var.set("● TRAINED")
            self.ml_badge.config(fg="#69ff47")
        else:
            self.ml_status_var.set("● NOT TRAINED")
            self.ml_badge.config(fg="#ff5252")

    # ── log helpers ───────────────────────────────────────────

    def _clear_log(self):
        def _do():
            self.log_box.config(state=tk.NORMAL)
            self.log_box.delete("1.0", tk.END)
            self.log_box.config(state=tk.DISABLED)
        self.root.after(0, _do)

    def _log(self, msg, tag="INFO"):
        def _do():
            self.log_box.config(state=tk.NORMAL)
            t = tag
            if t == "INFO":
                for c in ["ML","CSP","PLAN","FUZZY","SEARCH","RESCUE",
                          "KPI","EVENT","REPLAN","SYS","ERR"]:
                    if f"[{c}]" in msg:
                        t = c; break
            self.log_box.insert(tk.END, msg+"\n", t)
            self.log_box.see(tk.END)
            self.log_box.config(state=tk.DISABLED)
        self.root.after(0, _do)