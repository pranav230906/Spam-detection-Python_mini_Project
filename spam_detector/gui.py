# Placeholder for gui.py
# gui.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import pandas as pd
import os
import joblib
import datetime
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from utils import load_dataset
from pipeline_builder import get_pipeline, clean_text
from trainer import train_and_evaluate, plot_confusion_matrix_fig, plot_roc_curve_fig, get_top_features
from db import Database

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

class SpamDetectorApp:
    def __init__(self, master):
        self.master = master
        self.df = None
        self.pipeline = None
        self.current_metrics = None

        self.db = Database(os.path.join(DATA_DIR, "messages.db"))
        self._build_ui()

    def _build_ui(self):
        nb = ttk.Notebook(self.master)
        nb.pack(fill="both", expand=True)

        # Dataset tab
        self.tab_data = ttk.Frame(nb)
        nb.add(self.tab_data, text="Dataset")

        # Training tab
        self.tab_train = ttk.Frame(nb)
        nb.add(self.tab_train, text="Train & Evaluate")

        # Predict tab
        self.tab_predict = ttk.Frame(nb)
        nb.add(self.tab_predict, text="Predict")

        # DB tab
        self.tab_db = ttk.Frame(nb)
        nb.add(self.tab_db, text="Saved Messages (DB)")

        self._build_dataset_tab()
        self._build_train_tab()
        self._build_predict_tab()
        self._build_db_tab()

    # ---------------- Dataset tab ----------------
    def _build_dataset_tab(self):
        frame = self.tab_data
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill="x", padx=8, pady=8)

        ttk.Button(btn_frame, text="Load dataset", command=self.on_load_dataset).pack(side="left")
        ttk.Button(btn_frame, text="Load sample (data/spam.csv)", command=lambda: self.on_load_dataset(default_path=os.path.join(DATA_DIR, "spam.csv"))).pack(side="left", padx=6)
        ttk.Button(btn_frame, text="Summary", command=self.show_summary).pack(side="left", padx=6)

        # Table preview
        self.tree = ttk.Treeview(frame, columns=("label","text"), show="headings", height=12)
        self.tree.heading("label", text="Label")
        self.tree.heading("text", text="Text")
        self.tree.column("label", width=80, anchor="center")
        self.tree.column("text", width=900, anchor="w")
        self.tree.pack(fill="both", expand=True, padx=8, pady=8)

        # summary label
        self.summary_var = tk.StringVar()
        ttk.Label(frame, textvariable=self.summary_var).pack(fill="x", padx=8, pady=(0,8))

    def on_load_dataset(self, default_path=None):
        path = None
        if default_path and os.path.exists(default_path):
            path = default_path
        else:
            path = filedialog.askopenfilename(title="Open dataset", filetypes=[("CSV/TSV/TXT","*.csv *.tsv *.txt"),("All files","*.*")])
            if not path:
                return
        try:
            df = load_dataset(path)
        except Exception as e:
            messagebox.showerror("Error loading dataset", str(e))
            return
        self.df = df
        self._refresh_table()
        self.show_summary()
        messagebox.showinfo("Loaded", f"Loaded dataset with {len(self.df)} rows.")

    def _refresh_table(self, n=100):
        self.tree.delete(*self.tree.get_children())
        if self.df is None:
            return
        for _, row in self.df.head(n).iterrows():
            self.tree.insert("", "end", values=(row['label'], row['text']))

    def show_summary(self):
        if self.df is None:
            messagebox.showwarning("No dataset", "Load a dataset first.")
            return
        total = len(self.df)
        counts = self.df['label'].value_counts().to_dict()
        avg_len = int(self.df['text'].str.len().mean())
        self.summary_var.set(f"Rows: {total}  |  Counts: {counts}  |  Avg length: {avg_len}")

    # ---------------- Train tab ----------------
    def _build_train_tab(self):
        frame = self.tab_train
        left = ttk.Frame(frame)
        left.pack(side="left", fill="y", padx=8, pady=8)

        ttk.Label(left, text="Model:").pack(anchor="w")
        self.model_var = tk.StringVar(value="MultinomialNB")
        ttk.Combobox(left, textvariable=self.model_var, values=["MultinomialNB","LogisticRegression"], state="readonly").pack(fill="x")

        ttk.Label(left, text="Test size (0-1):").pack(anchor="w", pady=(10,0))
        self.test_size_var = tk.DoubleVar(value=0.2)
        ttk.Entry(left, textvariable=self.test_size_var).pack(fill="x")

        ttk.Label(left, text="Use GridSearch:").pack(anchor="w", pady=(10,0))
        self.grid_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, variable=self.grid_var).pack(anchor="w")

        ttk.Button(left, text="Train", command=self.on_train_clicked).pack(fill="x", pady=(12,4))
        ttk.Button(left, text="Save pipeline", command=self.on_save_pipeline).pack(fill="x")
        ttk.Button(left, text="Load pipeline", command=self.on_load_pipeline).pack(fill="x", pady=(4,0))

        self.train_log = tk.Text(left, height=20, width=40)
        self.train_log.pack(fill="both", expand=True, pady=(8,0))

        # Right side for plots
        right = ttk.Frame(frame)
        right.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        # Two canvases: confusion matrix and ROC
        self.fig_conf, self.ax_conf = plt.subplots(figsize=(4,3))
        self.canvas_conf = FigureCanvasTkAgg(self.fig_conf, master=right)
        self.canvas_conf.get_tk_widget().pack(side="top", fill="both", expand=True)

        self.fig_roc, self.ax_roc = plt.subplots(figsize=(4,3))
        self.canvas_roc = FigureCanvasTkAgg(self.fig_roc, master=right)
        self.canvas_roc.get_tk_widget().pack(side="top", fill="both", expand=True)

        # Top features
        self.fig_feat, self.ax_feat = plt.subplots(figsize=(6,2))
        self.canvas_feat = FigureCanvasTkAgg(self.fig_feat, master=right)
        self.canvas_feat.get_tk_widget().pack(side="top", fill="both", expand=True)

    def log(self, txt):
        self.train_log.insert("end", f"{txt}\n")
        self.train_log.see("end")

    def on_train_clicked(self):
        if self.df is None:
            messagebox.showwarning("No dataset", "Load a dataset first.")
            return
        t = threading.Thread(target=self._train_worker, daemon=True)
        t.start()

    def _train_worker(self):
        self._set_training_state(True)
        try:
            model_name = self.model_var.get()
            test_size = float(self.test_size_var.get())
            use_grid = bool(self.grid_var.get())
            self.log("Starting training...")
            pipeline, metrics, figs = train_and_evaluate(self.df, model_name=model_name, test_size=test_size, use_grid=use_grid)
            self.pipeline = pipeline
            self.current_metrics = metrics
            # show metrics in log
            self.log(f"Metrics: accuracy={metrics['accuracy']:.4f}, precision={metrics.get('precision',0):.4f}, recall={metrics.get('recall',0):.4f}, f1={metrics.get('f1',0):.4f}, auc={metrics.get('auc','N/A')}")
            # update plots
            conf_fig = figs.get('confusion')
            roc_fig = figs.get('roc')
            feat_fig = figs.get('features')

            if conf_fig:
                self._update_canvas_from_fig(conf_fig, self.canvas_conf, self.fig_conf)
            if roc_fig:
                self._update_canvas_from_fig(roc_fig, self.canvas_roc, self.fig_roc)
            if feat_fig:
                self._update_canvas_from_fig(feat_fig, self.canvas_feat, self.fig_feat)
        except Exception as e:
            self.log(f"Training error: {e}")
            messagebox.showerror("Training error", str(e))
        finally:
            self._set_training_state(False)

    def _update_canvas_from_fig(self, src_fig, canvas, target_fig):
        # copy artists from src_fig to target_fig
        target_fig.clf()
        # draw src to canvas buffer and reattach
        src_axes = src_fig.get_axes()
        for ax in src_axes:
            # create a new axis in target with same position
            newax = target_fig.add_axes(ax.get_position())
            for line in ax.get_lines():
                newax._add_line(line)
            for im in ax.get_images():
                newax.imshow(im.get_array(), aspect='auto', origin='lower')
            # copy text and patches manually (simple approach: re-draw by calling same plotting functions would be better)
            newax.set_title(ax.get_title())
            newax.set_xlabel(ax.get_xlabel())
            newax.set_ylabel(ax.get_ylabel())
        canvas.draw()

    def _set_training_state(self, running):
        state = "disabled" if running else "normal"
        # disable controls while training
        for child in self.tab_train.winfo_children():
            child.configure(state=state) if isinstance(child, ttk.Button) else None
        self.log("Training started..." if running else "Training complete.")

    def on_save_pipeline(self):
        if self.pipeline is None:
            messagebox.showwarning("No pipeline", "Train a model first.")
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".joblib", initialdir=MODELS_DIR, filetypes=[("Joblib","*.joblib")])
        if not save_path:
            return
        joblib.dump(self.pipeline, save_path)
        messagebox.showinfo("Saved", f"Pipeline saved to {save_path}")

    def on_load_pipeline(self):
        path = filedialog.askopenfilename(initialdir=MODELS_DIR, filetypes=[("Joblib","*.joblib"),("All","*.*")])
        if not path:
            return
        try:
            self.pipeline = joblib.load(path)
            messagebox.showinfo("Loaded", f"Pipeline loaded from {path}")
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    # ---------------- Predict tab ----------------
    def _build_predict_tab(self):
        frame = self.tab_predict
        top = ttk.Frame(frame)
        top.pack(fill="x", padx=8, pady=8)
        ttk.Label(top, text="Paste message to classify:").pack(anchor="w")
        self.predict_text = tk.Text(top, height=6)
        self.predict_text.pack(fill="x")

        btn_frame = ttk.Frame(top)
        btn_frame.pack(fill="x", pady=6)
        ttk.Button(btn_frame, text="Predict", command=self.on_predict).pack(side="left")
        ttk.Button(btn_frame, text="Save to DB", command=self.on_save_prediction).pack(side="left", padx=6)

        self.pred_result_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.pred_result_var, font=("TkDefaultFont", 12, "bold")).pack(anchor="w", pady=(6,0))

    def on_predict(self):
        txt = self.predict_text.get("1.0", "end").strip()
        if not txt:
            messagebox.showwarning("Empty", "Enter message text first.")
            return
        if self.pipeline is None:
            messagebox.showwarning("No model", "Train or load a model first.")
            return
        proba = None
        pred_label = None
        try:
            probs = self.pipeline.predict_proba([txt])
            if probs.shape[1] == 2:
                proba = float(max(probs[0]))
            else:
                proba = float(max(probs[0]))
            pred_label = self.pipeline.predict([txt])[0]
            self.pred_result_var.set(f"Predicted: {pred_label}  (conf: {proba:.3f})")
        except Exception as e:
            messagebox.showerror("Prediction error", str(e))

    def on_save_prediction(self):
        txt = self.predict_text.get("1.0", "end").strip()
        if not txt:
            messagebox.showwarning("Empty", "Enter message text first.")
            return
        if self.pipeline is None:
            messagebox.showwarning("No model", "Train or load a model first.")
            return
        true_label = simple_input_dialog(self.master, "Optional true label (leave blank if unknown):")
        probs = self.pipeline.predict_proba([txt])
        pred_label = self.pipeline.predict([txt])[0]
        prob = float(max(probs[0]))
        model_name = getattr(self.pipeline.named_steps['clf'], '__class__', type(self.pipeline.named_steps['clf'])).__name__
        self.db.insert_message(text=txt, true_label=true_label if true_label else None, predicted_label=pred_label, predicted_prob=prob, model=model_name)
        messagebox.showinfo("Saved", "Saved message to database.")
        self.refresh_db_table()

    # ---------------- DB tab ----------------
    def _build_db_tab(self):
        frame = self.tab_db
        top = ttk.Frame(frame)
        top.pack(fill="x", padx=8, pady=8)
        ttk.Button(top, text="Refresh", command=self.refresh_db_table).pack(side="left")
        ttk.Button(top, text="Show only spam", command=lambda: self.refresh_db_table(filter_label="spam")).pack(side="left", padx=6)
        ttk.Button(top, text="Show only ham", command=lambda: self.refresh_db_table(filter_label="ham")).pack(side="left", padx=6)
        ttk.Button(top, text="Export CSV", command=self.export_db).pack(side="left", padx=6)

        cols = ("id","text","true_label","predicted_label","predicted_prob","model","timestamp")
        self.db_tree = ttk.Treeview(frame, columns=cols, show="headings")
        for c in cols:
            self.db_tree.heading(c, text=c)
            self.db_tree.column(c, width=100 if c=="text" else 80, anchor="w")
        self.db_tree.pack(fill="both", expand=True, padx=8, pady=8)
        self.refresh_db_table()

    def refresh_db_table(self, filter_label=None):
        rows = self.db.query_messages(filter_label=filter_label)
        self.db_tree.delete(*self.db_tree.get_children())
        for r in rows:
            self.db_tree.insert("", "end", values=r)

    def export_db(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv")])
        if not path:
            return
        rows = self.db.query_messages()
        import csv
        with open(path, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id","text","true_label","predicted_label","predicted_prob","model","timestamp"])
            writer.writerows(rows)
        messagebox.showinfo("Exported", f"Exported {len(rows)} rows to {path}")


def simple_input_dialog(master, prompt):
    # blocking dialog for single-line input
    dlg = tk.Toplevel(master)
    dlg.title("Input")
    tk.Label(dlg, text=prompt).pack(padx=8, pady=8)
    ent = tk.Entry(dlg, width=60)
    ent.pack(padx=8, pady=(0,8))
    res = {"value": None}
    def ok():
        res["value"] = ent.get().strip()
        dlg.destroy()
    def cancel():
        dlg.destroy()
    ttk.Button(dlg, text="OK", command=ok).pack(side="left", padx=8, pady=8)
    ttk.Button(dlg, text="Cancel", command=cancel).pack(side="left", padx=8, pady=8)
    dlg.grab_set()
    dlg.wait_window()
    return res["value"]
