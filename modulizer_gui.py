#!/usr/bin/env python3
"""
Simple desktop UI for modulizer.
"""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
import importlib.util
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import typer

HERE = Path(__file__).resolve().parent
LOCAL_MODULIZER_PATH = HERE / "modulizer.py"
if not LOCAL_MODULIZER_PATH.exists():
    raise FileNotFoundError(f"Could not find local modulizer.py at {LOCAL_MODULIZER_PATH}")

spec = importlib.util.spec_from_file_location("modulizer_gui_local_modulizer", LOCAL_MODULIZER_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Failed to load modulizer module from {LOCAL_MODULIZER_PATH}")
modulizer = importlib.util.module_from_spec(spec)
if modulizer is None:
    raise ImportError(f"Failed to create module object from spec for {LOCAL_MODULIZER_PATH}")

sys.modules[spec.name] = modulizer
try:
    spec.loader.exec_module(modulizer)
except Exception:
    del sys.modules[spec.name]
    raise


class ModulizerGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Modulizer")
        self.root.geometry("980x980")

        self.input_file = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.created_location = tk.StringVar(value="Not created yet.")
        self.entrypoint_module = tk.StringVar(value="Not detected yet.")
        self.run_command = tk.StringVar(value="Run a modularization first.")
        self.architecture_mode = tk.StringVar(value="Not detected yet.")
        self.model = tk.StringVar(value=modulizer.LLMPlanner.DEFAULT_MODEL)
        self.planning_mode = tk.StringVar(value="safe")
        self.api_key = tk.StringVar()
        self.openai_base_url = tk.StringVar(value=modulizer.LLMPlanner.DEFAULT_BASE_URL)
        self.semantic_keywords = tk.StringVar()
        self.temperature = tk.DoubleVar(value=0.9)
        self.top_p = tk.DoubleVar(value=0.3)
        self.top_k = tk.IntVar(value=20)
        self.frequency_penalty = tk.DoubleVar(value=0.8)

        self.offline = tk.BooleanVar(value=False)
        self.verbose = tk.BooleanVar(value=False)
        self.strict_validation = tk.BooleanVar(value=False)
        self.semantic_grouping = tk.BooleanVar(value=True)

        self.max_modules = tk.IntVar(value=12)
        self.min_segments_per_module = tk.IntVar(value=3)
        self.ai_retries = tk.IntVar(value=5)
        self.heuristic_fallback = tk.BooleanVar(value=True)

        self.status_text = tk.StringVar(value="Ready.")
        self.last_manifest_path: Path | None = None
        self.last_output_dir: Path | None = None
        self._is_running = False
        self._build_ui()

    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 6}
        frm = ttk.Frame(self.root)
        frm.pack(fill="both", expand=True, padx=10, pady=10)
        frm.columnconfigure(1, weight=1)

        intro = ttk.Label(
            frm,
            text=(
                "Choose a Python file, pick where the modules should be created, "
                "then run. The generated modules folder is shown below and can be opened directly."
            ),
            wraplength=900,
            justify="left",
        )
        intro.grid(row=0, column=0, columnspan=3, sticky="ew", **pad)

        ttk.Label(frm, text="Input Python file").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.input_file, width=72).grid(row=1, column=1, sticky="ew", **pad)
        ttk.Button(frm, text="Browse", command=self._pick_input).grid(row=1, column=2, **pad)

        ttk.Label(frm, text="Output directory").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.output_dir, width=72).grid(row=2, column=1, sticky="ew", **pad)
        ttk.Button(frm, text="Browse", command=self._pick_output).grid(row=2, column=2, **pad)

        output_hint = ttk.Label(
            frm,
            text="This folder becomes the generated Python package. Example: C:\\project\\bot_modules",
            foreground="#555555",
        )
        output_hint.grid(row=3, column=1, columnspan=2, sticky="w", padx=8, pady=(0, 6))

        location_frame = ttk.LabelFrame(frm, text="Generated Modules Location")
        location_frame.grid(row=4, column=0, columnspan=3, sticky="ew", padx=8, pady=8)
        location_frame.columnconfigure(0, weight=1)
        ttk.Entry(
            location_frame,
            textvariable=self.created_location,
            state="readonly",
        ).grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        ttk.Button(location_frame, text="Open Folder", command=self._open_created_location).grid(row=0, column=1, padx=8, pady=8)
        ttk.Button(location_frame, text="Copy Path", command=self._copy_created_location).grid(row=0, column=2, padx=8, pady=8)

        run_frame = ttk.LabelFrame(frm, text="Run Info")
        run_frame.grid(row=5, column=0, columnspan=3, sticky="ew", padx=8, pady=8)
        run_frame.columnconfigure(1, weight=1)
        ttk.Label(run_frame, text="Entrypoint").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(run_frame, textvariable=self.entrypoint_module, state="readonly").grid(row=0, column=1, sticky="ew", padx=8, pady=6)
        ttk.Label(run_frame, text="Architecture").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(run_frame, textvariable=self.architecture_mode, state="readonly").grid(row=1, column=1, sticky="ew", padx=8, pady=6)
        ttk.Label(run_frame, text="Run command").grid(row=2, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(run_frame, textvariable=self.run_command, state="readonly").grid(row=2, column=1, sticky="ew", padx=8, pady=6)
        run_btns = ttk.Frame(run_frame)
        run_btns.grid(row=2, column=2, padx=8, pady=6)
        ttk.Button(run_btns, text="Copy Run Command", command=self._copy_run_command).pack(side="left")

        ttk.Label(frm, text="Model").grid(row=6, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.model).grid(row=6, column=1, columnspan=2, sticky="ew", **pad)

        ttk.Label(frm, text="Planning mode").grid(row=7, column=0, sticky="w", **pad)
        planning_combo = ttk.Combobox(
            frm,
            textvariable=self.planning_mode,
            values=["safe", "hybrid", "ai_first"],
            state="readonly",
        )
        planning_combo.grid(row=7, column=1, sticky="w", **pad)
        planning_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_planning_mode_changed())
        ttk.Label(
            frm,
            text="Safe = heuristics only, Hybrid = heuristics first with AI assist, AI-first = experimental.",
            foreground="#555555",
        ).grid(row=7, column=2, sticky="w", padx=8, pady=6)

        ttk.Label(frm, text="API key (optional if OPENAI_API_KEY is set)").grid(row=8, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.api_key, show="*").grid(row=8, column=1, columnspan=2, sticky="ew", **pad)

        ttk.Label(frm, text="API base URL (optional)").grid(row=9, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.openai_base_url).grid(row=9, column=1, columnspan=2, sticky="ew", **pad)

        numeric = ttk.LabelFrame(frm, text="Planning Controls")
        numeric.grid(row=10, column=0, columnspan=3, sticky="ew", padx=8, pady=10)
        numeric.columnconfigure(1, weight=1)
        numeric.columnconfigure(3, weight=1)
        numeric.columnconfigure(5, weight=1)

        ttk.Label(numeric, text="Max modules").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Spinbox(numeric, from_=1, to=64, textvariable=self.max_modules, width=8).grid(row=0, column=1, sticky="w")

        ttk.Label(numeric, text="Min segments/module").grid(row=0, column=2, sticky="w", padx=8, pady=6)
        ttk.Spinbox(numeric, from_=1, to=20, textvariable=self.min_segments_per_module, width=8).grid(row=0, column=3, sticky="w")

        ttk.Label(numeric, text="AI retries").grid(row=0, column=4, sticky="w", padx=8, pady=6)
        ttk.Spinbox(numeric, from_=1, to=15, textvariable=self.ai_retries, width=8).grid(row=0, column=5, sticky="w")

        ttk.Label(numeric, text="Semantic keywords (comma-separated)").grid(row=1, column=0, columnspan=2, sticky="w", padx=8, pady=6)
        ttk.Entry(numeric, textvariable=self.semantic_keywords).grid(row=1, column=2, columnspan=4, sticky="ew", padx=8, pady=6)

        ttk.Label(
            numeric,
            text="Tip: for large files, higher max modules usually gives better splits.",
            foreground="#555555",
        ).grid(row=2, column=0, columnspan=6, sticky="w", padx=8, pady=(0, 6))

        advanced = ttk.LabelFrame(frm, text="Advanced AI Settings")
        advanced.grid(row=11, column=0, columnspan=3, sticky="ew", padx=8, pady=8)
        advanced.columnconfigure(1, weight=1)
        advanced.columnconfigure(3, weight=1)
        advanced.columnconfigure(5, weight=1)

        ttk.Label(advanced, text="Temperature").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Spinbox(advanced, from_=0.0, to=2.0, increment=0.1, textvariable=self.temperature, width=8).grid(row=0, column=1, sticky="w")

        ttk.Label(advanced, text="Top p").grid(row=0, column=2, sticky="w", padx=8, pady=6)
        ttk.Spinbox(advanced, from_=0.0, to=1.0, increment=0.1, textvariable=self.top_p, width=8).grid(row=0, column=3, sticky="w")

        ttk.Label(advanced, text="Top k").grid(row=0, column=4, sticky="w", padx=8, pady=6)
        ttk.Spinbox(advanced, from_=1, to=100, textvariable=self.top_k, width=8).grid(row=0, column=5, sticky="w")

        ttk.Label(advanced, text="Frequency penalty").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        ttk.Spinbox(advanced, from_=-2.0, to=2.0, increment=0.1, textvariable=self.frequency_penalty, width=8).grid(row=1, column=1, sticky="w")

        toggles = ttk.Frame(frm)
        toggles.grid(row=12, column=0, columnspan=3, sticky="w", padx=8, pady=6)
        ttk.Checkbutton(toggles, text="Offline mode", variable=self.offline).grid(row=0, column=0, padx=6)
        ttk.Checkbutton(toggles, text="Strict validation", variable=self.strict_validation).grid(row=0, column=1, padx=6)
        ttk.Checkbutton(toggles, text="Semantic grouping", variable=self.semantic_grouping).grid(row=0, column=2, padx=6)
        ttk.Checkbutton(toggles, text="Verbose", variable=self.verbose).grid(row=0, column=3, padx=6)
        ttk.Checkbutton(
            toggles,
            text="Allow heuristic if AI fails",
            variable=self.heuristic_fallback,
        ).grid(row=0, column=4, padx=6)

        actions = ttk.Frame(frm)
        actions.grid(row=13, column=0, columnspan=3, sticky="ew", padx=8, pady=8)
        self.run_btn = ttk.Button(actions, text="Run Modularizer", command=self._start_run)
        self.run_btn.pack(side="left")
        ttk.Button(actions, text="Use Suggested Output Folder", command=self._apply_suggested_output_dir).pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="Quit", command=self.root.destroy).pack(side="right")

        cmd_frame = ttk.LabelFrame(frm, text="Commands (CLI)")
        cmd_frame.grid(row=14, column=0, columnspan=3, sticky="ew", padx=8, pady=8)
        cmd_frame.columnconfigure(0, weight=1)
        ttk.Label(
            cmd_frame,
            text="Equivalent terminal commands (run from the folder that contains modulizer.py). API key is not copied—use OPENAI_API_KEY or add --api-key manually.",
            wraplength=820,
        ).grid(row=0, column=0, columnspan=3, sticky="ew", padx=8, pady=(6, 4))

        self.cmd_text = tk.Text(cmd_frame, wrap="word", height=10, font=("Consolas", 9) if self._has_font("Consolas") else ("Courier New", 9))
        self.cmd_text.grid(row=1, column=0, columnspan=3, sticky="ew", padx=8, pady=4)

        cmd_btns = ttk.Frame(cmd_frame)
        cmd_btns.grid(row=2, column=0, columnspan=3, sticky="ew", padx=8, pady=(4, 8))
        ttk.Button(cmd_btns, text="Refresh commands", command=self._refresh_commands_text).pack(side="left", padx=(0, 8))
        ttk.Button(cmd_btns, text="Copy modularize command", command=self._copy_modularize_command).pack(side="left", padx=(0, 8))
        ttk.Button(cmd_btns, text="Copy all", command=self._copy_all_commands).pack(side="left")

        self._refresh_commands_text()

        log_frame = ttk.LabelFrame(frm, text="Log")
        log_frame.grid(row=15, column=0, columnspan=3, sticky="nsew", padx=8, pady=8)
        frm.rowconfigure(15, weight=1)

        self.log = tk.Text(log_frame, wrap="word", height=18)
        self.log.pack(fill="both", expand=True, padx=6, pady=6)

        status = ttk.Label(frm, textvariable=self.status_text, anchor="w")
        status.grid(row=16, column=0, columnspan=3, sticky="ew", padx=8, pady=6)
        self._on_planning_mode_changed()

    @staticmethod
    def _has_font(name: str) -> bool:
        try:
            import tkinter.font as tkfont

            return name.lower() in {f.lower() for f in tkfont.families()}
        except Exception:
            return False

    def _pick_input(self) -> None:
        chosen = filedialog.askopenfilename(
            title="Select Python file",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")],
        )
        if chosen:
            self.input_file.set(chosen)
            self._apply_suggested_output_dir(force=not self.output_dir.get().strip())
            self._refresh_commands_text()

    def _pick_output(self) -> None:
        chosen = filedialog.askdirectory(title="Select output directory")
        if chosen:
            self.output_dir.set(chosen)
            self._set_created_location(Path(chosen))
            self._refresh_commands_text()

    def _suggest_output_dir(self) -> Path | None:
        raw_input = self.input_file.get().strip()
        if not raw_input:
            return None
        input_path = Path(raw_input)
        stem = input_path.stem or "modulized_output"
        return input_path.parent / f"{stem}_modules"

    def _apply_suggested_output_dir(self, force: bool = True) -> None:
        suggested = self._suggest_output_dir()
        if suggested is None:
            return
        if force or not self.output_dir.get().strip():
            self.output_dir.set(str(suggested))
            self._set_created_location(suggested)
            self._refresh_commands_text()

    def _append_log(self, message: str) -> None:
        self.log.insert("end", message + "\n")
        self.log.see("end")

    def _set_created_location(self, path: Path | None) -> None:
        if path is None:
            self.created_location.set("Not created yet.")
            return
        self.created_location.set(str(path.resolve()))

    def _copy_created_location(self) -> None:
        current = self.created_location.get().strip()
        if not current or current == "Not created yet.":
            messagebox.showinfo("No folder yet", "Run the modularizer first, or choose an output directory.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(current)
        self.root.update_idletasks()
        self.status_text.set("Copied generated modules location.")

    def _copy_run_command(self) -> None:
        command = self.run_command.get().strip()
        if not command or command == "Run a modularization first.":
            messagebox.showinfo("No run command yet", "Run the modularizer first so the GUI can detect the generated entrypoint.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(command)
        self.root.update_idletasks()
        self.status_text.set("Copied run command.")

    def _open_created_location(self) -> None:
        raw = self.created_location.get().strip()
        if not raw or raw == "Not created yet.":
            messagebox.showinfo("No folder yet", "Run the modularizer first, or choose an output directory.")
            return
        path = Path(raw)
        target = path if path.exists() else path.parent
        if not target.exists():
            messagebox.showerror("Missing folder", f"Folder does not exist yet:\n{path}")
            return
        try:
            os.startfile(str(target))
        except Exception as exc:
            messagebox.showerror("Open failed", f"Could not open folder:\n{exc}")

    def _set_running(self, running: bool) -> None:
        self._is_running = running
        self.run_btn.configure(state="disabled" if running else "normal")
        if running:
            self.status_text.set("Running modularizer...")
        else:
            self.status_text.set("Ready.")

    def _on_planning_mode_changed(self) -> None:
        mode = self.planning_mode.get().strip().lower()
        if mode == "safe":
            self.offline.set(True)
            self.heuristic_fallback.set(True)
        elif mode == "hybrid":
            self.offline.set(False)
            self.heuristic_fallback.set(True)
        else:
            self.offline.set(False)
        self._refresh_commands_text()

    def _modularize_argv(self) -> list[str]:
        inp = self.input_file.get().strip()
        out = self.output_dir.get().strip()
        args: list[str] = [
            sys.executable,
            str(LOCAL_MODULIZER_PATH),
            "modularize",
            "--input-file",
            inp or ".",
            "--output-dir",
            out or ".",
        ]
        model = self.model.get().strip()
        if model:
            args.extend(["--model", model])
        args.extend(["--planning-mode", self.planning_mode.get().strip() or "safe"])
        if self.offline.get():
            args.append("--offline")
        base = self.openai_base_url.get().strip()
        if base:
            args.extend(["--openai-base-url", base])
        args.extend(["--temperature", str(self.temperature.get())])
        args.extend(["--top-p", str(self.top_p.get())])
        args.extend(["--top-k", str(self.top_k.get())])
        args.extend(["--frequency-penalty", str(self.frequency_penalty.get())])
        args.extend(["--max-modules", str(self.max_modules.get())])
        args.extend(["--min-segments-per-module", str(self.min_segments_per_module.get())])
        args.extend(["--ai-retries", str(self.ai_retries.get())])
        if self.semantic_grouping.get():
            args.append("--semantic-grouping")
        else:
            args.append("--no-semantic-grouping")
        if self.strict_validation.get():
            args.append("--strict-validation")
        else:
            args.append("--no-strict-validation")
        if self.verbose.get():
            args.append("--verbose")
        kw = self.semantic_keywords.get().strip()
        if kw:
            args.extend(["--semantic-keywords", kw])
        if self.heuristic_fallback.get():
            args.append("--heuristic-fallback")
        return args

    def _modularize_command_line(self) -> str:
        return subprocess.list2cmdline(self._modularize_argv())

    def _full_commands_text(self) -> str:
        lines = [
            "# Modularize (matches current UI options)",
            self._modularize_command_line(),
            "",
            "# Optional: create a JSON config template",
            subprocess.list2cmdline([sys.executable, str(LOCAL_MODULIZER_PATH), "init-config", "--output-file", "modulizer_config.json"]),
            "",
            "# Modularize using that config",
            subprocess.list2cmdline(
                [
                    sys.executable,
                    str(LOCAL_MODULIZER_PATH),
                    "modularize",
                    "--input-file",
                    self.input_file.get().strip() or "your_script.py",
                    "--output-dir",
                    self.output_dir.get().strip() or "output_modules",
                    "--config",
                    "modulizer_config.json",
                ]
            ),
            "",
            "# Show version",
            subprocess.list2cmdline(["python", "modulizer.py", "version"]),
            "",
            "# API key (PowerShell) — omit --api-key on the command line",
            r"# $env:OPENAI_API_KEY = 'your-key-here'",
        ]
        return "\n".join(lines)

    def _refresh_commands_text(self) -> None:
        self.cmd_text.configure(state="normal")
        self.cmd_text.delete("1.0", "end")
        self.cmd_text.insert("1.0", self._full_commands_text())
        self.cmd_text.configure(state="disabled")

    def _copy_modularize_command(self) -> None:
        cmd = self._modularize_command_line()
        self.root.clipboard_clear()
        self.root.clipboard_append(cmd)
        self.root.update_idletasks()
        self.status_text.set("Copied modularize command to clipboard.")

    def _copy_all_commands(self) -> None:
        text = self._full_commands_text()
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update_idletasks()
        self.status_text.set("Copied all commands to clipboard.")

    def _start_run(self) -> None:
        if self._is_running:
            return
        if not self.input_file.get().strip():
            messagebox.showerror("Missing input", "Please select an input Python file.")
            return
        if not self.output_dir.get().strip():
            messagebox.showerror("Missing output", "Please select an output directory.")
            return
        output_path = Path(self.output_dir.get().strip())
        self.last_output_dir = output_path
        self.last_manifest_path = output_path / "module_plan.json"
        self._set_created_location(output_path)
        self._set_running(True)
        self._append_log("-" * 72)
        self._append_log("Starting modularization...")
        self._append_log(f"Planning mode: {self.planning_mode.get().strip() or 'safe'}")
        self._append_log(f"Modules will be created in: {output_path}")
        thread = threading.Thread(target=self._run_modularize, daemon=True)
        thread.start()

    def _run_modularize(self) -> None:
        try:
            keywords = self.semantic_keywords.get().strip()
            old_echo = typer.echo

            def gui_echo(message: object = "", *args: object, **kwargs: object) -> None:
                text = str(message)
                self.root.after(0, lambda t=text: self._append_log(t))
                old_echo(message, *args, **kwargs)

            typer.echo = gui_echo
            try:
                modulizer.modularize(
                    input_file=Path(self.input_file.get().strip()),
                    output_dir=Path(self.output_dir.get().strip()),
                    model=self.model.get().strip() or None,
                    planning_mode=self.planning_mode.get().strip() or "safe",
                    api_key=self.api_key.get().strip() or None,
                    offline=self.offline.get(),
                    openai_base_url=self.openai_base_url.get().strip() or None,
                    temperature=self.temperature.get(),
                    top_p=self.top_p.get(),
                    top_k=self.top_k.get(),
                    frequency_penalty=self.frequency_penalty.get(),
                    config=None,
                    max_modules=self.max_modules.get(),
                    min_segments_per_module=self.min_segments_per_module.get(),
                    semantic_grouping=self.semantic_grouping.get(),
                    semantic_keywords=keywords if keywords else None,
                    strict_validation=self.strict_validation.get(),
                    ai_retries=self.ai_retries.get(),
                    heuristic_fallback=self.heuristic_fallback.get(),
                    verbose=self.verbose.get(),
                )
            finally:
                typer.echo = old_echo
            self.root.after(0, self._handle_success)
        except typer.Exit as e:
            code = getattr(e, "exit_code", 1)
            self.root.after(0, lambda c=code: self._append_log(f"Failed with exit code {c}."))
            self.root.after(0, lambda c=code: messagebox.showerror("Failed", f"Run failed with exit code {c}."))
        except Exception as e:
            self.root.after(0, lambda msg=str(e): self._append_log(f"Error: {msg}"))
            self.root.after(0, lambda msg=str(e): messagebox.showerror("Error", msg))
        finally:
            self.root.after(0, lambda: self._set_running(False))

    def _handle_success(self) -> None:
        output_dir = Path(self.output_dir.get().strip())
        self.last_output_dir = output_dir
        self.last_manifest_path = output_dir / "module_plan.json"
        self._set_created_location(output_dir)
        self._update_run_info()
        self._append_log("Done. Modularization completed.")
        self._append_log(f"Generated modules location: {output_dir}")

        manifest_note = ""
        if self.last_manifest_path.exists():
            manifest_note = f"\nManifest: {self.last_manifest_path}"
            self._append_log(f"Manifest written to: {self.last_manifest_path}")

        entrypoint_note = self.entrypoint_module.get().strip()
        architecture_note = self.architecture_mode.get().strip()
        run_note = self.run_command.get().strip()
        if entrypoint_note:
            self._append_log(f"Detected entrypoint: {entrypoint_note}")
        if architecture_note:
            self._append_log(f"Architecture path: {architecture_note}")
        if run_note and run_note != "Run a modularization first.":
            self._append_log(f"Recommended run command: {run_note}")

        self.status_text.set(f"Completed. Modules created in {output_dir}")
        messagebox.showinfo(
            "Success",
            f"Modularization completed successfully.\n\nModules created in:\n{output_dir}\n\nEntrypoint:\n{entrypoint_note}\n\nRun command:\n{run_note}{manifest_note}",
        )

    def _update_run_info(self) -> None:
        output_dir = self.last_output_dir or Path(self.output_dir.get().strip())
        manifest_path = self.last_manifest_path or (output_dir / "module_plan.json")
        package_name = output_dir.name.strip() or "modules"

        entrypoint = "Not detected"
        architecture = "Standard module split"
        run_command = "Run a modularization first."

        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                module_names = [
                    str(module.get("name", "")).strip()
                    for module in manifest.get("modules", [])
                    if isinstance(module, dict)
                ]
                notes = str(manifest.get("notes", "") or "")

                if "Promoted runtime architecture" in notes:
                    architecture = "Runtime-core architecture"
                if "main" in module_names:
                    entrypoint = "main"
                elif "shared" in module_names:
                    entrypoint = "shared"
                elif module_names:
                    entrypoint = module_names[0]

                if entrypoint != "Not detected":
                    run_command = f'python -m {package_name}.{entrypoint}'
            except Exception:
                entrypoint = "Not detected"
                architecture = "Could not read manifest"
                run_command = "Check module_plan.json manually."

        self.entrypoint_module.set(entrypoint)
        self.architecture_mode.set(architecture)
        self.run_command.set(run_command)


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    app = ModulizerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
