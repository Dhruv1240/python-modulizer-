#!/usr/bin/env python3
"""
Simple desktop UI for modulizer.
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import typer

import modulizer


class ModulizerGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Modulizer UI")
        self.root.geometry("900x920")

        self.input_file = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.model = tk.StringVar(value="gemini-2.5-flash")
        self.api_key = tk.StringVar()
        self.openai_base_url = tk.StringVar(value="https://generativelanguage.googleapis.com/v1beta/openai")
        self.semantic_keywords = tk.StringVar()

        self.offline = tk.BooleanVar(value=False)
        self.verbose = tk.BooleanVar(value=True)
        self.strict_validation = tk.BooleanVar(value=True)
        self.semantic_grouping = tk.BooleanVar(value=True)

        self.max_modules = tk.IntVar(value=8)
        self.min_segments_per_module = tk.IntVar(value=2)
        self.ai_retries = tk.IntVar(value=5)
        self.heuristic_fallback = tk.BooleanVar(value=False)

        self.status_text = tk.StringVar(value="Ready.")
        self._is_running = False
        self._build_ui()

    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 6}
        frm = ttk.Frame(self.root)
        frm.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Label(frm, text="Input Python file").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.input_file, width=72).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(frm, text="Browse", command=self._pick_input).grid(row=0, column=2, **pad)

        ttk.Label(frm, text="Output directory").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.output_dir, width=72).grid(row=1, column=1, sticky="ew", **pad)
        ttk.Button(frm, text="Browse", command=self._pick_output).grid(row=1, column=2, **pad)

        ttk.Label(frm, text="Model").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.model).grid(row=2, column=1, columnspan=2, sticky="ew", **pad)

        ttk.Label(frm, text="API key (optional if OPENAI_API_KEY is set)").grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.api_key, show="*").grid(row=3, column=1, columnspan=2, sticky="ew", **pad)

        ttk.Label(frm, text="API base URL (optional)").grid(row=4, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.openai_base_url).grid(row=4, column=1, columnspan=2, sticky="ew", **pad)

        numeric = ttk.LabelFrame(frm, text="Planning Controls")
        numeric.grid(row=5, column=0, columnspan=3, sticky="ew", padx=8, pady=10)
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

        toggles = ttk.Frame(frm)
        toggles.grid(row=6, column=0, columnspan=3, sticky="w", padx=8, pady=6)
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
        actions.grid(row=7, column=0, columnspan=3, sticky="ew", padx=8, pady=8)
        self.run_btn = ttk.Button(actions, text="Run Modularizer", command=self._start_run)
        self.run_btn.pack(side="left")
        ttk.Button(actions, text="Quit", command=self.root.destroy).pack(side="right")

        cmd_frame = ttk.LabelFrame(frm, text="Commands (CLI)")
        cmd_frame.grid(row=8, column=0, columnspan=3, sticky="ew", padx=8, pady=8)
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
        log_frame.grid(row=9, column=0, columnspan=3, sticky="nsew", padx=8, pady=8)
        frm.rowconfigure(9, weight=1)
        frm.columnconfigure(1, weight=1)

        self.log = tk.Text(log_frame, wrap="word", height=18)
        self.log.pack(fill="both", expand=True, padx=6, pady=6)

        status = ttk.Label(frm, textvariable=self.status_text, anchor="w")
        status.grid(row=10, column=0, columnspan=3, sticky="ew", padx=8, pady=6)

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
            if not self.output_dir.get():
                self.output_dir.set(str(Path(chosen).parent / "modulized_output"))
            self._refresh_commands_text()

    def _pick_output(self) -> None:
        chosen = filedialog.askdirectory(title="Select output directory")
        if chosen:
            self.output_dir.set(chosen)
            self._refresh_commands_text()

    def _append_log(self, message: str) -> None:
        self.log.insert("end", message + "\n")
        self.log.see("end")

    def _set_running(self, running: bool) -> None:
        self._is_running = running
        self.run_btn.configure(state="disabled" if running else "normal")
        self.status_text.set("Running..." if running else "Ready.")

    def _modularize_argv(self) -> list[str]:
        inp = self.input_file.get().strip()
        out = self.output_dir.get().strip()
        args: list[str] = [
            "python",
            "modulizer.py",
            "modularize",
            "--input-file",
            inp or ".",
            "--output-dir",
            out or ".",
        ]
        model = self.model.get().strip()
        if model:
            args.extend(["--model", model])
        if self.offline.get():
            args.append("--offline")
        base = self.openai_base_url.get().strip()
        if base:
            args.extend(["--openai-base-url", base])
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
            subprocess.list2cmdline(["python", "modulizer.py", "init-config", "--output-file", "modulizer_config.json"]),
            "",
            "# Modularize using that config",
            subprocess.list2cmdline(
                [
                    "python",
                    "modulizer.py",
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
        self._set_running(True)
        self._append_log("-" * 72)
        self._append_log("Starting modularization...")
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
                    api_key=self.api_key.get().strip() or None,
                    offline=self.offline.get(),
                    openai_base_url=self.openai_base_url.get().strip() or None,
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
            self.root.after(0, lambda: self._append_log("Done. Modularization completed."))
            self.root.after(0, lambda: messagebox.showinfo("Success", "Modularization completed successfully."))
        except typer.Exit as e:
            code = getattr(e, "exit_code", 1)
            self.root.after(0, lambda c=code: self._append_log(f"Failed with exit code {c}."))
            self.root.after(0, lambda c=code: messagebox.showerror("Failed", f"Run failed with exit code {c}."))
        except Exception as e:
            self.root.after(0, lambda msg=str(e): self._append_log(f"Error: {msg}"))
            self.root.after(0, lambda msg=str(e): messagebox.showerror("Error", msg))
        finally:
            self.root.after(0, lambda: self._set_running(False))


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    app = ModulizerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
