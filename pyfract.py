#!/usr/bin/env python3
"""
Modularize a large Python file into cohesive modules with (optional) AI assistance.
Uses the OpenAI Python SDK (openai.OpenAI) for chat completions when not in offline mode.
"""
# gawk gawk gawk meoww 

from __future__ import annotations

import ast
import importlib
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import typer
from typer.models import OptionInfo
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

app = typer.Typer(
    help="Split a large Python file into AI-planned modules.",
    add_completion=False,
)


@dataclass
class SymbolInfo:
    """Tracks a symbol's type, scope, and usage context."""
    name: str
    kind: str  # 'function', 'class', 'variable', 'method', etc.
    scope: str  # 'module' or 'local' or 'class'
    defined_at_line: int
    is_builtin: bool = False


@dataclass
class Segment:
    identifier: str
    kind: str
    name: str
    start_line: int
    end_line: int
    code: str
    signature: str
    dependencies: List[str]  # Names of module-level symbols this depends on
    # All names this top-level segment binds (constants in block_* segments, imports, defs, etc.)
    defined_symbols: List[str] = field(default_factory=list)
    local_symbols: List[str] = field(default_factory=list)  # Local vars, nested functions
    external_refs: Dict[str, str] = field(default_factory=dict)  # name -> symbol_kind
    used_attributes: List[Tuple[str, str]] = field(default_factory=list)  # (obj, attr) pairs


class SourceAnalyzer:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.builtins = {
            'print', 'len', 'str', 'int', 'dict', 'list', 'set', 'tuple',
            'open', 'close', 'range', 'enumerate', 'zip', 'map', 'filter',
            'True', 'False', 'None', 'self', 'cls', 'object', 'Exception',
            'property', 'staticmethod', 'classmethod', '__name__', '__main__'
        }
        self.module_symbols: Dict[str, SymbolInfo] = {}  # Module-level symbols

    def analyze(self) -> Tuple[str, List[Segment]]:
        source = self.path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        segments: List[Segment] = []
        lines = source.splitlines()

        # First pass: collect all module-level symbols
        self._collect_module_symbols(tree)

        # Second pass: analyze segments
        for node in tree.body:
            if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
                continue
            start = self._segment_start_line(node)
            end = node.end_lineno
            code = "\n".join(lines[start - 1 : end])
            kind, name = self._classify(node)
            identifier = f"{kind}:{name}:{start}"
            signature = self._signature(node)
            defined_symbols = self._defined_symbols_for_top_level_node(node)
            
            # Enhanced dependency analysis with scope awareness
            dependencies, local_symbols, external_refs, used_attributes = self._analyze_dependencies(node)
            
            segments.append(
                Segment(
                    identifier=identifier,
                    kind=kind,
                    name=name,
                    start_line=start,
                    end_line=end,
                    code=code,
                    signature=signature,
                    dependencies=dependencies,
                    defined_symbols=defined_symbols,
                    local_symbols=local_symbols,
                    external_refs=external_refs,
                    used_attributes=used_attributes,
                )
            )

        summary = (
            f"{self.path.name} | {len(lines)} lines | "
            f"{len(segments)} top-level segments detected."
        )
        return summary, segments

    def _collect_module_symbols(self, tree: ast.AST) -> None:
        """First pass: collect all module-level symbol definitions."""
        for node in tree.body:
            defined_names = self._defined_symbols_for_top_level_node(node)
            if not defined_names:
                continue

            if isinstance(node, ast.FunctionDef):
                kind = 'function'
            elif isinstance(node, ast.AsyncFunctionDef):
                kind = 'async_function'
            elif isinstance(node, ast.ClassDef):
                kind = 'class'
            else:
                kind = 'variable'

            for name in defined_names:
                self.module_symbols[name] = SymbolInfo(
                    name=name, kind=kind, scope='module', defined_at_line=getattr(node, 'lineno', 0)
                )

    @staticmethod
    def _segment_start_line(node: ast.AST) -> int:
        start = getattr(node, "lineno", 1)
        decorators = getattr(node, "decorator_list", None)
        if decorators:
            decorator_lines = [
                getattr(decorator, "lineno", start)
                for decorator in decorators
                if hasattr(decorator, "lineno")
            ]
            if decorator_lines:
                start = min(start, min(decorator_lines))
        return start

    @staticmethod
    def _classify(node: ast.AST) -> Tuple[str, str]:
        if isinstance(node, ast.FunctionDef):
            return "function", node.name
        if isinstance(node, ast.AsyncFunctionDef):
            return "async_function", node.name
        if isinstance(node, ast.ClassDef):
            return "class", node.name
        end_lineno = getattr(node, "end_lineno", node.lineno)
        return "block", f"block_{node.lineno}_{end_lineno}"

    @staticmethod
    def _assignment_target_names(target: ast.AST) -> List[str]:
        if isinstance(target, ast.Name):
            return [target.id]
        if isinstance(target, (ast.Tuple, ast.List)):
            out: List[str] = []
            for el in target.elts:
                out.extend(SourceAnalyzer._assignment_target_names(el))
            return out
        if isinstance(target, ast.Starred):
            return SourceAnalyzer._assignment_target_names(target.value)
        return []

    @staticmethod
    def _defined_symbols_for_top_level_node(node: ast.AST) -> List[str]:
        """Names bound at module scope by this top-level AST node (including compound bodies)."""
        names: List[str] = []

        def from_stmt(stmt: ast.stmt) -> None:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.append(stmt.name)
                return
            if isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    names.extend(SourceAnalyzer._assignment_target_names(t))
                return
            if isinstance(stmt, ast.AnnAssign):
                if isinstance(stmt.target, ast.Name):
                    names.append(stmt.target.id)
                return
            if isinstance(stmt, ast.AugAssign):
                if isinstance(stmt.target, ast.Name):
                    names.append(stmt.target.id)
                return
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    bound = alias.asname if alias.asname else alias.name.split(".")[0]
                    names.append(bound)
                return
            if isinstance(stmt, ast.ImportFrom):
                for alias in stmt.names:
                    if alias.name == "*":
                        continue
                    bound = alias.asname if alias.asname else alias.name
                    names.append(bound)
                return
            if isinstance(stmt, ast.If):
                for s in stmt.body:
                    from_stmt(s)
                for s in stmt.orelse:
                    from_stmt(s)
                return
            if isinstance(stmt, (ast.For, ast.AsyncFor)):
                names.extend(SourceAnalyzer._assignment_target_names(stmt.target))
                for s in stmt.body:
                    from_stmt(s)
                for s in stmt.orelse:
                    from_stmt(s)
                return
            if isinstance(stmt, ast.While):
                for s in stmt.body:
                    from_stmt(s)
                for s in stmt.orelse:
                    from_stmt(s)
                return
            if isinstance(stmt, (ast.With, ast.AsyncWith)):
                for item in stmt.items:
                    if item.optional_vars:
                        names.extend(SourceAnalyzer._assignment_target_names(item.optional_vars))
                for s in stmt.body:
                    from_stmt(s)
                return
            if isinstance(stmt, ast.Try):
                for s in stmt.body:
                    from_stmt(s)
                for handler in stmt.handlers:
                    if handler.name:
                        names.append(handler.name)
                    for s in handler.body:
                        from_stmt(s)
                for s in stmt.orelse:
                    from_stmt(s)
                for s in stmt.finalbody:
                    from_stmt(s)
                return
            match_cls = getattr(ast, "Match", None)
            if match_cls is not None and isinstance(stmt, match_cls):
                for case in stmt.cases:
                    for s in case.body:
                        from_stmt(s)

        if isinstance(node, ast.stmt):
            from_stmt(node)
        return list(dict.fromkeys(names))

    @staticmethod
    def _signature(node: ast.AST) -> str:
        try:
            return ast.unparse(node)[:500]  # type: ignore[attr-defined]
        except Exception:
            return node.__class__.__name__

    def _analyze_dependencies(self, node: ast.AST) -> Tuple[List[str], List[str], Dict[str, str], List[Tuple[str, str]]]:
        """
        Enhanced dependency analysis with:
        - Scope awareness (local vs module)
        - Symbol type tracking (function, class, variable)
        - Attribute filtering (self.x, obj.y)
        - No false imports from local variables
        
        Returns:
        - dependencies: List[str] of module-level symbols used
        - local_symbols: List[str] of locally defined symbols
        - external_refs: Dict[str, str] of (name -> kind) for used symbols
        - used_attributes: List[Tuple[str, str]] of (object, attribute) accesses
        """
        local_scope = LocalScopeAnalyzer()
        dependency_collector = DependencyCollector(self.module_symbols, local_scope)
        
        dependency_collector.visit(node)
        
        # Filter: only include module-level symbols
        real_dependencies = []
        for name in dependency_collector.referenced_names:
            if name in self.builtins:
                continue
            if name in local_scope.all_local_symbols:
                continue  # Local variable, not a cross-module dependency
            if name in self.module_symbols:
                real_dependencies.append(name)
        
        return (
            sorted(real_dependencies),
            local_scope.all_local_symbols,
            {name: self.module_symbols.get(name, SymbolInfo(name, 'unknown', 'module', 0)).kind 
             for name in real_dependencies},
            dependency_collector.attribute_accesses,
        )


class LocalScopeAnalyzer(ast.NodeVisitor):
    """Identifies all locally-defined symbols (local vars, nested functions, etc.)."""
    def __init__(self):
        self.all_local_symbols: List[str] = []
        self.scope_stack: List[Set[str]] = [set()]  # Stack of scopes
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Add function parameters to local scope
        for arg in node.args.args:
            self.all_local_symbols.append(arg.arg)
        for arg in node.args.posonlyargs:
            self.all_local_symbols.append(arg.arg)
        for arg in node.args.kwonlyargs:
            self.all_local_symbols.append(arg.arg)
        if node.args.vararg:
            self.all_local_symbols.append(node.args.vararg.arg)
        if node.args.kwarg:
            self.all_local_symbols.append(node.args.kwarg.arg)
        self.generic_visit(node)
    
    visit_AsyncFunctionDef = visit_FunctionDef
    
    def visit_Assign(self, node: ast.Assign) -> None:
        # Track local assignments
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.all_local_symbols.append(target.id)
        self.generic_visit(node)


class DependencyCollector(ast.NodeVisitor):
    """Collects dependencies while tracking scope and filtering attributes."""
    def __init__(self, module_symbols: Dict[str, SymbolInfo], local_scope: LocalScopeAnalyzer):
        self.module_symbols = module_symbols
        self.local_scope = local_scope
        self.referenced_names: Set[str] = set()
        self.attribute_accesses: List[Tuple[str, str]] = []
        self.in_attribute = False
    
    def visit_Call(self, node: ast.Call) -> None:
        # Function call: func() or obj.method()
        if isinstance(node.func, ast.Name):
            self.referenced_names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                # Track attribute access but don't treat attribute name as dependency
                obj_name = node.func.value.id
                self.referenced_names.add(obj_name)
                self.attribute_accesses.append((obj_name, node.func.attr))
        self.generic_visit(node)
    
    def visit_Name(self, node: ast.Name) -> None:
        # Only track names being loaded, not stored
        if isinstance(node.ctx, ast.Load):
            self.referenced_names.add(node.id)
        self.generic_visit(node)
    
    def visit_Attribute(self, node: ast.Attribute) -> None:
        # object.attribute access - track object, NOT attribute
        if isinstance(node.value, ast.Name):
            self.referenced_names.add(node.value.id)
            self.attribute_accesses.append((node.value.id, node.attr))
        self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Base classes
        for base in node.bases:
            if isinstance(base, ast.Name):
                self.referenced_names.add(base.id)
        self.generic_visit(node)
    
    def visit_arg(self, node: ast.arg) -> None:
        # Type annotations in function arguments
        if node.annotation:
            if isinstance(node.annotation, ast.Name):
                self.referenced_names.add(node.annotation.id)
        self.generic_visit(node)


class LLMPlanner:
    """Matches typical OpenAI-client usage (e.g. AIMLAPI at https://ai.aimlapi.com)."""

    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
    DEFAULT_MODEL = "gemini-2.5-flash"
    PLANNING_MODES = {"safe", "hybrid", "ai_first"}

    @staticmethod
    def _normalize_option(value: Any, default: Any) -> Any:
        if isinstance(value, OptionInfo):
            return default
        return value

    def __init__(
        self,
        model: str,
        api_key: Optional[str],
        temperature: float = 0.9,
        top_p: float = 0.3,
        top_k: int = 20,
        frequency_penalty: float = 0.8,
        offline: bool = False,
        max_retries: int = 5,
        max_modules: int = 8,
        min_segments_per_module: int = 2,
        semantic_grouping: bool = True,
        semantic_keywords: Optional[List[str]] = None,
        base_url: Optional[str] = None,
        verbose: bool = False,
        allow_heuristic_fallback: bool = False,
        planning_mode: str = "safe",
    ) -> None:
        self.model = self._normalize_option(model, self.DEFAULT_MODEL)
        self.temperature = float(self._normalize_option(temperature, 0.9))
        self.top_p = float(self._normalize_option(top_p, 0.3))
        self.top_k = int(self._normalize_option(top_k, 20))
        self.frequency_penalty = float(self._normalize_option(frequency_penalty, 0.8))
        self.offline = bool(self._normalize_option(offline, False))
        self.max_retries = max(1, int(self._normalize_option(max_retries, 5)))
        self.allow_heuristic_fallback = bool(
            self._normalize_option(allow_heuristic_fallback, False)
        )
        requested_mode = str(self._normalize_option(planning_mode, "safe") or "safe").strip().lower()
        aliases = {"ai-first": "ai_first", "ai first": "ai_first"}
        requested_mode = aliases.get(requested_mode, requested_mode)
        self.planning_mode = requested_mode if requested_mode in self.PLANNING_MODES else "safe"
        self.max_modules = max(1, int(self._normalize_option(max_modules, 8)))
        self.min_segments_per_module = max(1, int(self._normalize_option(min_segments_per_module, 2)))
        self.semantic_grouping = bool(self._normalize_option(semantic_grouping, True))
        self.semantic_keywords = [
            k.strip().lower()
            for k in (self._normalize_option(semantic_keywords, []) or [])
            if k and str(k).strip()
        ]
        self.verbose = bool(self._normalize_option(verbose, False))
        raw_base = (
            self._normalize_option(base_url, os.environ.get("OPENAI_BASE_URL") or self.DEFAULT_BASE_URL)
            or self.DEFAULT_BASE_URL
        ).rstrip("/")
        self.base_url = raw_base
        self.use_google_compat = "generativelanguage.googleapis.com" in self.base_url
        self._openai_client: Optional[OpenAI] = None

        needs_ai_client = self.planning_mode in {"hybrid", "ai_first"} and not self.offline
        if needs_ai_client:
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key and self.planning_mode == "ai_first":
                raise RuntimeError(
                    "Missing API key. Use --api-key or set OPENAI_API_KEY."
                )
            if key:
                self._openai_client = OpenAI(
                    base_url=raw_base,
                    api_key=key,
                    timeout=120.0,
                )

    @staticmethod
    def _message_text(message: Any) -> Optional[str]:
        """Normalize chat message.content (SDK object, dict, string, or parts) to plain text."""
        if message is None:
            return None
        content: Any
        if isinstance(message, dict):
            content = message.get("content")
        else:
            content = getattr(message, "content", None)
        if content is None:
            return None
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if hasattr(part, "text"):
                    parts.append(str(getattr(part, "text", "")))
                elif isinstance(part, dict):
                    if part.get("type") == "text" and "text" in part:
                        parts.append(str(part["text"]))
                    elif "text" in part:
                        parts.append(str(part["text"]))
                elif isinstance(part, str):
                    parts.append(part)
            return "".join(parts) if parts else None
        return str(content)

    @staticmethod
    def _detect_architecture_profile(metadata: List[Dict[str, Any]]) -> str:
        app_signals = 0
        cli_signals = 0
        class_count = 0

        app_terms = [
            "bot.run(",
            "commands.bot",
            "discord.intents",
            "on_ready",
            "on_message",
            "token",
            "fastapi(",
            "flask(",
            "@app.route",
            "@bot.event",
            "uvicorn.run",
        ]
        cli_terms = [
            "typer.typer",
            "@app.command",
            "argparse",
            "click.command",
            "parser.add_argument",
            "def modularize",
            "def main",
            "__main__",
            "init_config",
            "version",
        ]

        for entry in metadata:
            if entry.get("kind") == "class":
                class_count += 1
            text = " ".join(
                [
                    str(entry.get("name", "")),
                    str(entry.get("signature_excerpt", "")),
                    " ".join(
                        str(sym) for sym in entry.get("defined_symbols", []) if isinstance(sym, str)
                    ),
                ]
            ).lower()
            if any(term in text for term in app_terms):
                app_signals += 1
            if any(term in text for term in cli_terms):
                cli_signals += 1

        if app_signals >= 3 and app_signals >= cli_signals + 1:
            return "application_runtime"
        if cli_signals >= 2:
            return "tool_cli"
        if class_count >= 3:
            return "generic_library"
        return "generic_script"

    @staticmethod
    def _architecture_guidance(profile: str) -> str:
        if profile == "application_runtime":
            return (
                "Architecture target: application/runtime file. Prefer modules such as "
                "runtime_core, main, data_storage, analytics, visuals, battle, economy, progression, commands."
            )
        if profile == "tool_cli":
            return (
                "Architecture target: tool/CLI file. Prefer layered modules such as "
                "models_types, analysis_module, llmplanner_module, modulewriter_module, cli_module, shared_module. "
                "Do not force a runtime_core/main bot-style architecture."
            )
        if profile == "generic_library":
            return (
                "Architecture target: reusable library/module file. Prefer modules such as "
                "models, analysis, processing, io, validation, api, shared."
            )
        return (
            "Architecture target: generic Python module. Prefer cohesive, low-coupling modules and avoid monolithic output."
        )

    def plan(self, summary: str, segments: Sequence[Segment]) -> Dict[str, Any]:
        metadata = [
            {
                "segment_id": seg.identifier,
                "kind": seg.kind,
                "name": seg.name,
                "lines": f"{seg.start_line}-{seg.end_line}",
                "dependencies": seg.dependencies[:20],
                "defined_symbols": seg.defined_symbols,
                "signature_excerpt": seg.signature[:200],
            }
            for seg in segments
        ]
        architecture_profile = self._detect_architecture_profile(metadata)

        heuristic_plan = self._fallback_plan(
            metadata,
            max_modules=self.max_modules,
            min_segments_per_module=self.min_segments_per_module,
            semantic_grouping=self.semantic_grouping,
            semantic_keywords=self.semantic_keywords,
        )

        if self.offline or self.planning_mode == "safe":
            profile_label = architecture_profile.replace("_", " ")
            if self.offline:
                typer.echo(f"Offline mode enabled: using heuristic {profile_label} plan.", err=True)
            else:
                typer.echo(f"Safe planning mode enabled: using heuristic {profile_label} plan.", err=True)
            return heuristic_plan

        if self.planning_mode == "hybrid":
            if self._openai_client is None:
                typer.echo(
                    f"Hybrid planning selected but no API key/client is available. Using heuristic {architecture_profile.replace('_', ' ')} plan.",
                    err=True,
                )
                return heuristic_plan
            return self._plan_hybrid(summary, metadata, heuristic_plan)

        return self._plan_ai_first(summary, metadata, heuristic_plan)

    def _plan_hybrid(
        self,
        summary: str,
        metadata: List[Dict[str, Any]],
        heuristic_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        heur_modules = len(heuristic_plan.get("modules", []))
        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": "You improve module plans conservatively and return precise JSON only.",
            },
            {
                "role": "user",
                "content": textwrap.dedent(
                    f"""
                    Improve this existing heuristic module plan without breaking it.
                    File summary: {summary}
                    {self._architecture_guidance(architecture_profile)}

                    Segments (JSON):
                    {json.dumps(metadata, indent=2, default=str)}

                    Current heuristic plan (JSON):
                    {json.dumps(heuristic_plan, indent=2, default=str)}

                    Rules:
                    - Return valid JSON in the same format: {{"modules": [...], "notes": "..."}}
                    - Every listed segment_id must exist and appear exactly once.
                    - Do not invent segment_ids.
                    - Stay feature-oriented and conservative.
                    - Keep the plan at least as granular as the heuristic plan unless a merge is clearly necessary.
                    - Avoid one giant module.
                    - Prefer useful module names like bot_core, data_storage, moderation, analytics, visuals, battle, economy, progression, commands_general, aura_commands, shared.
                    - If the heuristic plan is already strong, you may keep it with only better naming/descriptions.

                    Constraints:
                    - target max modules: {self.max_modules}
                    - minimum segments per module: {self.min_segments_per_module}
                    - heuristic module count to preserve or improve: {heur_modules}
                    """
                ).strip(),
            },
        ]
        ai_plan = self._run_ai_planning_attempts(messages, metadata)
        if ai_plan is None:
            typer.echo("Hybrid planning kept the heuristic feature-based plan.", err=True)
            return heuristic_plan
        return ai_plan

    def _plan_ai_first(
        self,
        summary: str,
        metadata: List[Dict[str, Any]],
        heuristic_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": "You produce precise JSON and thoughtful module plans.",
            },
            {
                "role": "user",
                "content": textwrap.dedent(
                    f"""
                    You are a senior software architect. Refactor the following Python file into modules.
                    File summary: {summary}
                    {self._architecture_guidance(architecture_profile)}

                    Segments (JSON):
                    {json.dumps(metadata, indent=2, default=str)}

                    Return valid JSON with:
                    {{
                      "modules": [
                        {{
                          "name": "short_snake_case_name",
                          "description": "purpose of the module",
                          "segment_ids": ["segment_id", ...]
                        }}
                      ],
                      "shared_helpers": "notes about common utilities" (optional),
                      "notes": "extra instructions" (optional)
                    }}
                    Ensure every listed segment_id exists. Each segment_id must appear exactly once across all modules. Do not invent new segment_ids; use only the provided list. Avoid duplicates.
                    Prefer cohesive modules and avoid tiny modules unless unavoidable.
                    Avoid monolithic plans where one module contains most of the file.
                    For large bots or multi-feature files, strongly prefer feature-based modules such as:
                    - bot_core / events / config / data_storage
                    - commands_general / commands_admin / commands_economy / commands_analytics
                    - aura_core / moderation / visuals / cards / battles / tournaments / shop / leaderboards
                    Keep data-loading code and heavy visual generation separate from command handlers when possible.
                    Constraints:
                    - target max modules: {self.max_modules}
                    - minimum segments per module: {self.min_segments_per_module}
                    - semantic grouping enabled: {self.semantic_grouping}
                    - domain keywords: {self.semantic_keywords if self.semantic_keywords else "[]"}
                    """
                ).strip(),
            },
        ]
        ai_plan = self._run_ai_planning_attempts(messages, metadata)
        if ai_plan is not None:
            return ai_plan
        if self.allow_heuristic_fallback:
            typer.echo("AI-first planning failed. Falling back to heuristic feature-based plan.", err=True)
            return heuristic_plan
        raise RuntimeError("AI-first planning failed and heuristic fallback is disabled.")

    def _run_ai_planning_attempts(
        self,
        messages: List[Dict[str, str]],
        metadata: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if self._openai_client is None:
            return None

        client = self._openai_client

        def _chat_create() -> Any:
            # Match OpenAI SDK: client.chat.completions.create(..., top_k=...) as used by AIMLAPI.
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
            if not self.use_google_compat:
                kwargs["frequency_penalty"] = self.frequency_penalty

            try:
                if self.use_google_compat:
                    return client.chat.completions.create(**kwargs)
                return client.chat.completions.create(
                    **kwargs,
                    top_k=self.top_k,
                    response_format={"type": "json_object"},
                )
            except TypeError:
                try:
                    if self.use_google_compat:
                        return client.chat.completions.create(**kwargs)
                    return client.chat.completions.create(
                        **kwargs,
                        response_format={"type": "json_object"},
                        extra_body={"top_k": self.top_k},
                    )
                except TypeError:
                    return client.chat.completions.create(**kwargs)

        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = _chat_create()
                if self.verbose:
                    try:
                        raw_preview = response.model_dump_json()[:500]
                    except Exception:
                        raw_preview = str(response)[:500]
                    typer.echo(f"Chat completion raw response (attempt {attempt}):\n{raw_preview}", err=True)

                choices = getattr(response, "choices", None) or []
                if not choices:
                    last_error = "AI response missing choices"
                    continue

                message = choices[0].message
                content = self._message_text(message)
                if content is None:
                    last_error = "AI response content is null"
                    continue

                content = content.strip()
                if not content:
                    last_error = "AI response content is empty"
                    continue

                # Strip markdown code fences the model sometimes wraps JSON in
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)

                parsed_plan = json.loads(content)
                parsed_plan, sanitation_warnings = self._sanitize_ai_plan(parsed_plan, metadata)
                if sanitation_warnings and self.verbose:
                    for warning in sanitation_warnings:
                        typer.echo(f"AI plan warning: {warning}", err=True)

                parsed_plan = self._complete_missing_segments(parsed_plan, metadata)

                # STRICTLY validate AI output
                is_valid, error_msg = self._validate_ai_plan(parsed_plan, metadata)
                if not is_valid and error_msg.startswith("Circular dependencies detected:"):
                    merged_plan = self._merge_cyclic_plan(parsed_plan, metadata)
                    merged_valid, merged_error = self._validate_ai_plan(merged_plan, metadata)
                    if merged_valid:
                        granular_enough, granular_msg = self._is_plan_granular_enough(
                            merged_plan,
                            metadata,
                            requested_max_modules=self.max_modules,
                        )
                        if not granular_enough:
                            last_error = f"AI plan collapsed too aggressively after cycle merge: {granular_msg}"
                            typer.echo(
                                f"{last_error}. Falling back to heuristic feature-based planning.",
                                err=True,
                            )
                            return self._fallback_plan(
                                metadata,
                                max_modules=self.max_modules,
                                min_segments_per_module=self.min_segments_per_module,
                                semantic_grouping=self.semantic_grouping,
                                semantic_keywords=self.semantic_keywords,
                            )
                        if self.verbose:
                            typer.echo("AI plan contained module cycles; auto-merged cyclic modules.", err=True)
                        return merged_plan
                    last_error = f"AI plan validation failed after cycle merge: {merged_error}"
                    continue
                if not is_valid:
                    last_error = f"AI plan validation failed: {error_msg}"
                    continue

                return parsed_plan

            except APIStatusError as exc:
                last_error = f"API HTTP {getattr(exc, 'status_code', '?')}: {exc}"
                if self.verbose and getattr(exc, "response", None) is not None:
                    try:
                        typer.echo(exc.response.text[:500], err=True)
                    except Exception:
                        pass
            except (APIConnectionError, APITimeoutError, RateLimitError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            except (KeyError, IndexError, json.JSONDecodeError, AttributeError, TypeError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"

            if self.verbose:
                typer.echo(f"AI planning attempt {attempt}/{self.max_retries} failed: {last_error}", err=True)

        if self.verbose:
            typer.echo(f"AI planning exhausted retries: {last_error}", err=True)
        return None

    @staticmethod
    def _complete_missing_segments(plan: Dict[str, Any], metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
        """If the AI omits some segments, auto-assign them so planning can still succeed."""
        if not isinstance(plan, dict):
            return plan
        modules = plan.get("modules", [])
        if not isinstance(modules, list):
            return plan

        valid_segment_ids = {entry["segment_id"] for entry in metadata}
        assigned = {
            seg_id
            for m in modules
            if isinstance(m, dict)
            for seg_id in m.get("segment_ids", [])
            if isinstance(seg_id, str)
        }
        missing = sorted(list(valid_segment_ids - assigned))
        if not missing:
            return plan

        # Merge into a misc/utilities module if it exists; otherwise create one.
        preferred = {"utilities", "utilities_module", "misc", "misc_module"}
        target_idx: Optional[int] = None
        for i, m in enumerate(modules):
            if not isinstance(m, dict):
                continue
            name = str(m.get("name", "")).strip().lower()
            if name in preferred:
                target_idx = i
                break

        filled_module_name = ""
        if target_idx is None:
            existing = {str(m.get("name", "")).strip() for m in modules if isinstance(m, dict)}
            base = "utilities_module"
            name = base
            n = 1
            while name in existing or not name:
                name = f"{base}_{n}"
                n += 1
            filled_module_name = name
            modules.append(
                {
                    "name": name,
                    "description": f"Auto-filled module for {len(missing)} segments omitted by the AI plan.",
                    "segment_ids": missing,
                }
            )
        else:
            m = dict(modules[target_idx])
            filled_module_name = str(m.get("name", "utilities_module"))
            segs = list(m.get("segment_ids", []))
            seg_set = {s for s in segs if isinstance(s, str)}
            for mid in missing:
                if mid not in seg_set:
                    segs.append(mid)
            m["segment_ids"] = segs
            if not m.get("description"):
                m["description"] = "Contains miscellaneous segments (auto-filled)."
            modules[target_idx] = m

        plan["modules"] = modules
        typer.echo(
            f"AI plan omitted {len(missing)} segment(s); assigned them to module {filled_module_name!r}.",
            err=True,
        )
        return plan

    @staticmethod
    def _validate_ai_plan(plan: Dict[str, Any], metadata: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """
        STRICTLY validate AI-generated plan for correctness.
        Checks for: duplicates, orphans, empty modules, and circular dependencies.
        Returns: (is_valid, error_message)
        """
        if not isinstance(plan, dict):
            return False, "Plan is not a dictionary"
        
        if "modules" not in plan or not isinstance(plan.get("modules"), list):
            return False, "Missing or invalid 'modules' field"
        
        if len(plan["modules"]) == 0:
            return False, "Plan contains no modules"
        
        # Build set of valid segment IDs
        valid_segment_ids = {entry["segment_id"] for entry in metadata}
        all_assigned_segments = set()
        seen_names = set()
        
        # Validate each module
        for i, module in enumerate(plan["modules"]):
            if not isinstance(module, dict):
                return False, f"Module {i} is not a dict"
            
            # Check required fields
            if "name" not in module or not isinstance(module["name"], str) or not module["name"]:
                return False, f"Module {i} has missing or invalid name"
            
            if "segment_ids" not in module or not isinstance(module["segment_ids"], list):
                return False, f"Module {i} has missing or invalid segment_ids"
            
            # Check module is not empty
            if len(module["segment_ids"]) == 0:
                return False, f"Module {module['name']} is empty (no segments assigned)"
            
            # Check for duplicate module names
            if module["name"] in seen_names:
                return False, f"Duplicate module name: {module['name']}"
            seen_names.add(module["name"])
            
            # Validate segment IDs
            for seg_id in module["segment_ids"]:
                if seg_id not in valid_segment_ids:
                    return False, f"Module {i} references invalid segment: {seg_id}"
                
                # Check for segments assigned to multiple modules
                if seg_id in all_assigned_segments:
                    return False, f"Segment {seg_id} assigned to multiple modules (DUPLICATE!)"
                
                all_assigned_segments.add(seg_id)
        
        # Ensure ALL segments are assigned (NO ORPHANS)
        if all_assigned_segments != valid_segment_ids:
            missing = valid_segment_ids - all_assigned_segments
            return False, f"Segments not assigned to any module: {missing}"
        
        # Check for circular dependencies between modules
        module_deps = LLMPlanner._build_module_dependencies(plan, metadata)
        cycles = LLMPlanner._detect_cycles(module_deps)
        if cycles:
            cycle_info = ", ".join([" -> ".join(cycle) for cycle in cycles])
            return False, f"Circular dependencies detected: {cycle_info}"
        
        return True, ""

    @staticmethod
    def _sanitize_ai_plan(plan: Dict[str, Any], metadata: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[str]]:
        """Clean up recoverable issues in AI-generated plans."""
        warnings: List[str] = []
        if not isinstance(plan, dict):
            return plan, warnings

        modules = plan.get("modules")
        if not isinstance(modules, list):
            return plan, warnings

        valid_segment_ids = {entry["segment_id"] for entry in metadata}
        seen_segments: Set[str] = set()
        sanitized_modules: List[Dict[str, Any]] = []

        for module in modules:
            if not isinstance(module, dict):
                warnings.append("Removed invalid module entry that is not an object.")
                continue

            segment_ids = []
            for seg_id in module.get("segment_ids", []):
                if not isinstance(seg_id, str):
                    warnings.append(f"Ignored non-string segment_id {seg_id!r} in module {module.get('name', '<unknown>')}.")
                    continue
                if seg_id not in valid_segment_ids:
                    warnings.append(f"Ignored invalid segment_id {seg_id!r} in module {module.get('name', '<unknown>')}.")
                    continue
                if seg_id in seen_segments:
                    warnings.append(f"Removed duplicate segment_id {seg_id!r} from module {module.get('name', '<unknown>')}.")
                    continue
                segment_ids.append(seg_id)
                seen_segments.add(seg_id)

            if segment_ids:
                sanitized_module = dict(module)
                sanitized_module["segment_ids"] = segment_ids
                sanitized_modules.append(sanitized_module)
            else:
                warnings.append(f"Dropped module {module.get('name', '<unknown>')} because it contained no valid segments after sanitization.")

        # Missing segments are filled in by _complete_missing_segments() after sanitization;
        # do not warn here (avoids confusing "missing" messages right before auto-assignment).

        sanitized_plan = {**plan, "modules": sanitized_modules}
        return sanitized_plan, warnings

    @staticmethod
    def _build_module_dependencies(plan: Dict[str, Any], metadata: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
        """Build module-level dependency graph."""
        # Map segment_id -> module_name
        segment_to_module: Dict[str, str] = {}
        for module in plan.get("modules", []):
            for seg_id in module.get("segment_ids", []):
                segment_to_module[seg_id] = module["name"]

        # Map symbol name -> set(modules) (handles collisions conservatively)
        name_to_modules: Dict[str, Set[str]] = {}
        for entry in metadata:
            seg_id = entry.get("segment_id")
            module_name = segment_to_module.get(seg_id)
            if not module_name:
                continue
            defined = entry.get("defined_symbols")
            if isinstance(defined, list) and defined:
                for sym_name in defined:
                    if not isinstance(sym_name, str):
                        continue
                    name_to_modules.setdefault(sym_name, set()).add(module_name)
            else:
                # Fallback for legacy metadata without defined_symbols
                if isinstance(seg_id, str):
                    parts = seg_id.split(":")
                    if len(parts) >= 2:
                        name_to_modules.setdefault(parts[1], set()).add(module_name)

        # Map segment_id -> dependencies
        segment_deps: Dict[str, List[str]] = {}
        for entry in metadata:
            segment_deps[entry["segment_id"]] = entry.get("dependencies", [])
        
        # Build module-level dependencies
        module_deps: Dict[str, Set[str]] = {m["name"]: set() for m in plan.get("modules", [])}
        for seg_id, deps in segment_deps.items():
            source_module = segment_to_module.get(seg_id)
            if not source_module:
                continue
            
            for dep_name in deps:
                for target_module in name_to_modules.get(dep_name, set()):
                    if target_module != source_module:
                        module_deps[source_module].add(target_module)
        
        return module_deps

    @staticmethod
    def _strongly_connected_components(graph: Dict[str, Set[str]]) -> List[List[str]]:
        index: Dict[str, int] = {}
        lowlink: Dict[str, int] = {}
        stack: List[str] = []
        on_stack: Set[str] = set()
        result: List[List[str]] = []

        def strongconnect(node: str) -> None:
            index[node] = len(index)
            lowlink[node] = index[node]
            stack.append(node)
            on_stack.add(node)

            for neighbor in graph.get(node, set()):
                if neighbor not in index:
                    strongconnect(neighbor)
                    lowlink[node] = min(lowlink[node], lowlink[neighbor])
                elif neighbor in on_stack:
                    lowlink[node] = min(lowlink[node], index[neighbor])

            if lowlink[node] == index[node]:
                component: List[str] = []
                while True:
                    current = stack.pop()
                    on_stack.remove(current)
                    component.append(current)
                    if current == node:
                        break
                result.append(component)

        for node in graph:
            if node not in index:
                strongconnect(node)

        return result

    @staticmethod
    def _merge_cyclic_plan(plan: Dict[str, Any], metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
        modules = [m for m in plan.get("modules", []) if isinstance(m, dict)]
        if not modules:
            return plan

        module_deps = LLMPlanner._build_module_dependencies({"modules": modules}, metadata)
        sccs = LLMPlanner._strongly_connected_components(module_deps)
        cyclic_groups = [sorted(component) for component in sccs if len(component) > 1]
        if not cyclic_groups:
            return plan

        module_map = {str(module["name"]): module for module in modules if module.get("name")}
        merged_modules: List[Dict[str, Any]] = []
        merged_names: Set[str] = set()

        for cycle in cyclic_groups:
            merged_names.update(cycle)
            merged_segment_ids: List[str] = []
            seen_segments: Set[str] = set()
            descriptions: List[str] = []
            for module_name in cycle:
                module = module_map.get(module_name)
                if not module:
                    continue
                description = str(module.get("description", "")).strip()
                if description:
                    descriptions.append(description)
                for seg_id in module.get("segment_ids", []):
                    if isinstance(seg_id, str) and seg_id not in seen_segments:
                        seen_segments.add(seg_id)
                        merged_segment_ids.append(seg_id)

            merged_modules.append(
                {
                    "name": "_".join(cycle),
                    "description": " / ".join(dict.fromkeys(descriptions))
                    or f"Merged cyclic modules: {', '.join(cycle)}",
                    "segment_ids": merged_segment_ids,
                }
            )

        final_modules = [
            module for module in modules if str(module.get("name")) not in merged_names
        ]
        final_modules.extend(merged_modules)

        merged_plan = dict(plan)
        merged_plan["modules"] = final_modules
        merged_notes = str(plan.get("notes", "")).strip()
        merged_names_flat = ", ".join(
            sorted(name for group in cyclic_groups for name in group)
        )
        merged_plan["notes"] = (
            f"{merged_notes}\nAuto-merged cyclic modules during AI plan validation: {merged_names_flat}."
            if merged_notes
            else f"Auto-merged cyclic modules during AI plan validation: {merged_names_flat}."
        )
        return merged_plan

    @staticmethod
    def _is_plan_granular_enough(
        plan: Dict[str, Any],
        metadata: List[Dict[str, Any]],
        requested_max_modules: int,
    ) -> Tuple[bool, str]:
        modules = [m for m in plan.get("modules", []) if isinstance(m, dict)]
        total_segments = len(metadata)
        module_count = len(modules)
        if total_segments <= 0:
            return True, ""

        segment_counts = [
            len([seg for seg in module.get("segment_ids", []) if isinstance(seg, str)])
            for module in modules
        ]
        largest_module = max(segment_counts, default=0)
        largest_ratio = largest_module / total_segments

        if total_segments >= 80:
            minimum_useful_modules = min(max(4, requested_max_modules // 2), requested_max_modules)
            if module_count < minimum_useful_modules:
                return (
                    False,
                    f"only {module_count} modules remained for {total_segments} segments "
                    f"(expected at least {minimum_useful_modules})",
                )

            if largest_ratio > 0.55:
                return (
                    False,
                    f"largest module contains {largest_module}/{total_segments} segments "
                    f"({largest_ratio:.0%}), which is too monolithic",
                )

        return True, ""

    @staticmethod
    def _detect_cycles(graph: Dict[str, Set[str]]) -> List[List[str]]:
        """Detect cycles in dependency graph using DFS."""
        cycles = []
        visited = set()
        rec_stack = set()
        path = []
        
        def dfs(node: str) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)
            
            path.pop()
            rec_stack.remove(node)
        
        for node in graph:
            if node not in visited:
                dfs(node)
        
        return cycles

    @staticmethod
    def _fallback_plan(
        metadata: List[Dict[str, Any]],
        max_modules: int = 8,
        min_segments_per_module: int = 2,
        semantic_grouping: bool = True,
        semantic_keywords: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Intelligent heuristic planning that groups related code segments.
        Analyzes dependencies and relationships to create functional modules.
        """
        if not metadata:
            return {"modules": [], "notes": "No segments to process."}

        semantic_keywords = [k.strip().lower() for k in (semantic_keywords or []) if k.strip()]
        architecture_profile = LLMPlanner._detect_architecture_profile(metadata)

        if len(metadata) >= 80:
            if architecture_profile == "tool_cli":
                tool_plan = LLMPlanner._tool_cli_plan(
                    metadata=metadata,
                    max_modules=max_modules,
                    min_segments_per_module=min_segments_per_module,
                    semantic_keywords=semantic_keywords,
                )
                if tool_plan is not None:
                    return tool_plan
            if architecture_profile == "generic_library":
                library_plan = LLMPlanner._library_first_plan(
                    metadata=metadata,
                    max_modules=max_modules,
                    min_segments_per_module=min_segments_per_module,
                    semantic_keywords=semantic_keywords,
                )
                if library_plan is not None:
                    return library_plan
            feature_plan = LLMPlanner._feature_first_plan(
                metadata=metadata,
                max_modules=max_modules,
                min_segments_per_module=min_segments_per_module,
                semantic_keywords=semantic_keywords,
            )
            if feature_plan is not None:
                return feature_plan

        # Build segment records
        segments = []
        for entry in metadata:
            segments.append({
                "id": entry["segment_id"],
                "kind": entry["kind"],
                "name": entry["name"],
                "dependencies": set(entry.get("dependencies", [])),
                "line": int(entry["lines"].split("-")[0]) if "lines" in entry else 0,
                "tokens": set(re.findall(r"[a-zA-Z0-9]+", entry["name"].lower())),
            })

        # Add semantic tokens from user keywords.
        for seg in segments:
            for kw in semantic_keywords:
                if kw and kw in seg["name"].lower():
                    seg["tokens"].add(kw)

        # Build weighted affinity graph (predictive grouping)
        affinity_graph: Dict[str, Dict[str, int]] = {}
        for segment in segments:
            affinity_graph[segment["id"]] = {}

            # Find segments that this one depends on
            for other in segments:
                if other["id"] == segment["id"]:
                    continue

                score = 0

                # Strong edge for explicit dependency
                if other["name"] in segment["dependencies"]:
                    score += 4

                # Classes should include related top-level functions when they depend on the class
                if segment["kind"] == "class" and other["kind"] in {"function", "async_function"}:
                    if segment["name"] in other["dependencies"]:
                        score += 3
                    elif other["name"].startswith(segment["name"].lower() + "_") or \
                         other["name"].startswith(segment["name"] + "."):
                        score += 2

                # Semantic affinity from naming conventions
                if semantic_grouping:
                    shared = segment["tokens"].intersection(other["tokens"])
                    if shared:
                        score += min(2, len(shared))
                    seg_bucket = LLMPlanner._semantic_bucket(segment["name"])
                    other_bucket = LLMPlanner._semantic_bucket(other["name"])
                    if seg_bucket == other_bucket and seg_bucket != "other":
                        score += 1

                # Proximity signal: declarations near each other likely related
                if abs(segment["line"] - other["line"]) <= 40:
                    score += 1

                if score > 0:
                    affinity_graph[segment["id"]][other["id"]] = score

        # Predictive grouping using affinity expansion from strongest anchors
        visited = set()
        groups: List[List[str]] = []
        sorted_segments = sorted(
            segments,
            key=lambda s: (len(s["dependencies"]), s["kind"] == "class", -s["line"]),
            reverse=True,
        )

        target_modules = min(
            max_modules,
            max(1, len(segments) // min_segments_per_module),
        )

        for segment in sorted_segments:
            if segment["id"] in visited:
                continue

            group = set()
            stack = [segment["id"]]

            while stack:
                current_id = stack.pop()
                if current_id in visited:
                    continue

                visited.add(current_id)
                group.add(current_id)

                neighbors = affinity_graph.get(current_id, {})
                ranked_neighbors = sorted(neighbors.items(), key=lambda kv: kv[1], reverse=True)
                for neighbor_id, weight in ranked_neighbors:
                    if neighbor_id in visited:
                        continue
                    # Keep groups cohesive: avoid adding low-signal neighbor if group already viable
                    if len(group) >= min_segments_per_module and weight <= 1:
                        continue
                    stack.append(neighbor_id)

            if group:
                groups.append(sorted(list(group)))

        # If relationships are weak, use simple semantic grouping
        if len(groups) <= 1:
            groups = LLMPlanner._simple_grouping(segments, semantic_keywords)

        # Predictively normalize group count and size before creating modules
        groups = LLMPlanner._normalize_groups(
            groups=groups,
            segments=segments,
            target_modules=target_modules,
            min_segments_per_module=min_segments_per_module,
            semantic_keywords=semantic_keywords,
        )

        # Create modules from groups
        modules = []
        used_names = set()
        
        for idx, group_ids in enumerate(groups, start=1):
            # Find representative segment for naming
            group_segments = [s for s in segments if s["id"] in group_ids]
            if not group_segments:
                continue

            # Choose the most important segment for naming
            primary_segment = max(group_segments,
                                key=lambda s: (s["kind"] == "class", len(s["dependencies"]), s["name"]))

            # Create module name based on primary segment
            base_name = primary_segment["name"].split("_")[0] or primary_segment["kind"]
            module_name = f"{base_name}_module"
            
            # Ensure unique names
            counter = 1
            original_name = module_name
            while module_name in used_names:
                module_name = f"{original_name}_{counter}"
                counter += 1
            used_names.add(module_name)

            # Create description
            kinds = [s["kind"] for s in group_segments]
            kind_counts = {kind: kinds.count(kind) for kind in set(kinds)}
            description_parts = []
            for kind, count in kind_counts.items():
                description_parts.append(f"{count} {kind}{'s' if count > 1 else ''}")
            description = f"Contains {', '.join(description_parts)}"

            modules.append({
                "name": module_name,
                "description": description,
                "segment_ids": group_ids,
            })

        return {
            "modules": modules,
            "notes": (
                f"Generated via predictive heuristic planning with {len(groups)} cohesive groups "
                f"(max_modules={max_modules}, min_segments_per_module={min_segments_per_module}, "
                f"semantic_grouping={semantic_grouping})."
            ),
        }

    @staticmethod
    def _semantic_bucket(name: str) -> str:
        name_lower = name.lower()
        if any(word in name_lower for word in ["aura", "rank", "leaderboard", "score", "streak"]):
            return "aura"
        if any(word in name_lower for word in ["graph", "stats", "trend", "predict", "compare", "board", "analytics"]):
            return "analytics"
        if any(word in name_lower for word in ["card", "image", "draw", "theme", "badge", "avatar", "visual"]):
            return "visuals"
        if any(word in name_lower for word in ["battle", "duel", "fight", "stake", "shield"]):
            return "battle"
        if any(word in name_lower for word in ["shop", "inventory", "item", "equip", "buy"]):
            return "economy"
        if any(word in name_lower for word in ["tournament", "daily", "reward", "claim", "prize"]):
            return "progression"
        if any(word in name_lower for word in ["message", "toxic", "moderation", "semantic", "safe", "hate"]):
            return "moderation"
        if any(word in name_lower for word in ["discord", "bot", "ready", "event", "prefix", "command", "guild", "channel"]):
            return "bot"
        if any(word in name_lower for word in ["load", "save", "data", "file", "read", "write", "parse", "serialize"]):
            return "data"
        if any(word in name_lower for word in ["config", "setup", "init", "env", "settings"]):
            return "config"
        if any(word in name_lower for word in ["main", "run", "start", "entry", "cli"]):
            return "main"
        if any(word in name_lower for word in ["util", "helper", "common", "shared", "base"]):
            return "utils"
        if any(word in name_lower for word in ["model", "schema", "entity", "dto"]):
            return "domain"
        if any(word in name_lower for word in ["http", "api", "client", "request", "response"]):
            return "integration"
        return "other"

    @staticmethod
    def _feature_bucket(name: str, kind: str) -> str:
        name_lower = name.lower()

        if kind == "block":
            if any(word in name_lower for word in ["block_1_", "block_2_", "block_3_"]):
                return "bot_core"
            bucket = LLMPlanner._semantic_bucket(name_lower)
            if bucket == "data":
                return "data_storage"
            if bucket == "config":
                return "config"

        if any(word in name_lower for word in ["on_ready", "on_message", "prefix", "gate", "bot", "channel", "guild"]):
            return "bot_core"
        if any(word in name_lower for word in ["load_", "save_", "json", "data", "config", "file", "inventory"]):
            return "data_storage"
        if any(word in name_lower for word in ["graph", "stats", "trend", "predict", "compare", "board", "leaderboard", "servercompare"]):
            return "analytics"
        if any(word in name_lower for word in ["card", "image", "draw", "theme", "badge", "avatar", "visual", "progress_bar"]):
            return "visuals"
        if any(word in name_lower for word in ["battle", "duel", "stake", "shield"]):
            return "battle"
        if any(word in name_lower for word in ["shop", "inventory", "item", "equip", "buy"]):
            return "economy"
        if any(word in name_lower for word in ["tournament", "daily", "streak", "reward", "claim", "snapshot", "prize"]):
            return "progression"
        if any(word in name_lower for word in ["toxic", "semantic", "moderation", "safe", "hate", "hostile", "banter", "aura_local", "aura_rules"]):
            return "moderation"
        if any(word in name_lower for word in ["help", "invite", "debug", "reset", "enable", "disable", "status", "info", "ping", "commandlist"]):
            return "commands_general"
        if any(word in name_lower for word in ["slash", "aura", "rank", "auraof", "auraboard"]):
            return "aura_commands"

        return "shared"

    @staticmethod
    def _feature_first_plan(
        metadata: List[Dict[str, Any]],
        max_modules: int,
        min_segments_per_module: int,
        semantic_keywords: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        semantic_keywords = [k.strip().lower() for k in (semantic_keywords or []) if k.strip()]
        preferred_order = [
            "bot_core",
            "config",
            "data_storage",
            "moderation",
            "aura_commands",
            "commands_general",
            "analytics",
            "visuals",
            "battle",
            "economy",
            "progression",
            "shared",
        ]
        buckets: Dict[str, List[str]] = {name: [] for name in preferred_order}

        for entry in metadata:
            name = str(entry.get("name", ""))
            kind = str(entry.get("kind", ""))
            bucket = LLMPlanner._feature_bucket(name, kind)

            if semantic_keywords and bucket == "shared":
                lowered_name = name.lower()
                if any(keyword in lowered_name for keyword in semantic_keywords):
                    bucket = "commands_general"

            buckets.setdefault(bucket, []).append(entry["segment_id"])

        non_empty = {key: value for key, value in buckets.items() if value}
        if len(non_empty) < 4:
            return None

        groups = [segment_ids for _, segment_ids in non_empty.items()]
        target_modules = min(max_modules, max(1, len(metadata) // max(1, min_segments_per_module)))
        segments = [
            {
                "id": entry["segment_id"],
                "kind": entry["kind"],
                "name": entry["name"],
                "dependencies": set(entry.get("dependencies", [])),
                "line": int(entry["lines"].split("-")[0]) if "lines" in entry else 0,
                "tokens": set(re.findall(r"[a-zA-Z0-9]+", str(entry["name"]).lower())),
            }
            for entry in metadata
        ]

        groups = LLMPlanner._normalize_groups(
            groups=groups,
            segments=segments,
            target_modules=target_modules,
            min_segments_per_module=min_segments_per_module,
            semantic_keywords=semantic_keywords,
        )

        id_to_bucket = {}
        for bucket_name, segment_ids in non_empty.items():
            for seg_id in segment_ids:
                id_to_bucket[seg_id] = bucket_name

        modules: List[Dict[str, Any]] = []
        used_names: Set[str] = set()
        for group in groups:
            bucket_votes: Dict[str, int] = {}
            for seg_id in group:
                bucket = id_to_bucket.get(seg_id, "shared")
                bucket_votes[bucket] = bucket_votes.get(bucket, 0) + 1
            module_name = max(
                bucket_votes.items(),
                key=lambda item: (item[1], -preferred_order.index(item[0]) if item[0] in preferred_order else 0),
            )[0]

            base_name = module_name
            counter = 1
            while base_name in used_names:
                base_name = f"{module_name}_{counter}"
                counter += 1
            used_names.add(base_name)

            modules.append(
                {
                    "name": base_name,
                    "description": f"Feature-oriented module for {base_name.replace('_', ' ')}.",
                    "segment_ids": group,
                }
            )

        return {
            "modules": modules,
            "notes": (
                "Generated via feature-first heuristic planning for a large file; "
                "prioritized functional separation before dependency normalization."
            ),
        }

    @staticmethod
    def _tool_cli_bucket(name: str, kind: str, signature_excerpt: str) -> str:
        lowered = name.lower()
        text = f"{lowered} {signature_excerpt.lower()}"
        if kind == "class":
            if lowered in {"symbolinfo", "segment"} or "dataclass" in text:
                return "models_types"
            if any(token in lowered for token in ["analyzer", "collector", "parser", "resolver", "scope"]):
                return "analysis_module"
            if any(token in lowered for token in ["planner", "client"]):
                return "llmplanner_module"
            if any(token in lowered for token in ["writer", "builder", "exporter", "renderer"]):
                return "modulewriter_module"
            if any(token in lowered for token in ["validator", "checker", "verifier"]):
                return "validation_module"
        if any(token in text for token in ["@app.command", "typer.typer", "__main__", "version(", "init_config(", "modularize("]):
            return "cli_module"
        if any(token in lowered for token in ["analyze", "parse", "collect", "resolve", "dependency"]):
            return "analysis_module"
        if any(token in lowered for token in ["plan", "group", "sanitize", "cycle"]):
            return "llmplanner_module"
        if any(token in lowered for token in ["write", "import", "validate", "sort_modules"]):
            return "modulewriter_module"
        return "shared_module"

    @staticmethod
    def _tool_cli_plan(
        metadata: List[Dict[str, Any]],
        max_modules: int,
        min_segments_per_module: int,
        semantic_keywords: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        preferred_order = [
            "models_types",
            "analysis_module",
            "llmplanner_module",
            "modulewriter_module",
            "validation_module",
            "cli_module",
            "shared_module",
        ]
        buckets: Dict[str, List[str]] = {name: [] for name in preferred_order}
        for entry in metadata:
            bucket = LLMPlanner._tool_cli_bucket(
                name=str(entry.get("name", "")),
                kind=str(entry.get("kind", "")),
                signature_excerpt=str(entry.get("signature_excerpt", "")),
            )
            buckets.setdefault(bucket, []).append(entry["segment_id"])

        non_empty = {key: value for key, value in buckets.items() if value}
        if len(non_empty) < 4:
            return None

        ordered = [(name, list(ids)) for name, ids in buckets.items() if ids]
        normalized: List[Tuple[str, List[str]]] = ordered
        if len(normalized) > max_modules:
            while len(normalized) > max_modules and len(normalized) > 1:
                name, ids = normalized.pop()
                normalized[-1][1].extend(ids)

        modules = [
            {
                "name": name,
                "description": f"Tool/CLI architecture module for {name.replace('_', ' ')}.",
                "segment_ids": ids,
            }
            for name, ids in normalized
        ]
        return {
            "modules": modules,
            "notes": "Generated via tool/CLI heuristic planning with layered modules.",
        }

    @staticmethod
    def _library_role_bucket(name: str, kind: str, signature_excerpt: str) -> str:
        lowered = name.lower()
        text = f"{lowered} {signature_excerpt.lower()}"
        if kind == "class":
            if any(token in lowered for token in ["model", "schema", "config", "state", "info", "result"]):
                return "models"
            if any(token in lowered for token in ["analyzer", "parser", "collector", "resolver", "visitor"]):
                return "analysis"
            if any(token in lowered for token in ["writer", "reader", "loader", "saver", "serializer", "exporter"]):
                return "io"
            if any(token in lowered for token in ["validator", "checker", "verifier"]):
                return "validation"
            if any(token in lowered for token in ["manager", "service", "engine", "planner", "client"]):
                return "processing"
        if any(token in text for token in ["__main__", "@app.command", "typer.typer", "click.command", "argparse"]):
            return "api"
        if any(token in lowered for token in ["validate", "check", "verify"]):
            return "validation"
        if any(token in lowered for token in ["write", "read", "load", "save", "serialize", "export", "import"]):
            return "io"
        if any(token in lowered for token in ["analyze", "parse", "collect", "resolve"]):
            return "analysis"
        if any(token in lowered for token in ["plan", "process", "engine", "service", "manager"]):
            return "processing"
        return "shared"

    @staticmethod
    def _library_first_plan(
        metadata: List[Dict[str, Any]],
        max_modules: int,
        min_segments_per_module: int,
        semantic_keywords: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        preferred_order = ["models", "analysis", "processing", "io", "validation", "api", "shared"]
        buckets: Dict[str, List[str]] = {name: [] for name in preferred_order}
        for entry in metadata:
            bucket = LLMPlanner._library_role_bucket(
                name=str(entry.get("name", "")),
                kind=str(entry.get("kind", "")),
                signature_excerpt=str(entry.get("signature_excerpt", "")),
            )
            buckets.setdefault(bucket, []).append(entry["segment_id"])

        ordered = [(name, ids) for name, ids in buckets.items() if ids]
        if len(ordered) < 3:
            return None
        normalized: List[Tuple[str, List[str]]] = [(name, list(ids)) for name, ids in ordered]
        if len(normalized) > max_modules:
            while len(normalized) > max_modules and len(normalized) > 1:
                name, ids = normalized.pop()
                normalized[-1][1].extend(ids)

        modules = [
            {
                "name": name,
                "description": f"Library-oriented module for {name.replace('_', ' ')}.",
                "segment_ids": ids,
            }
            for name, ids in normalized
        ]
        return {
            "modules": modules,
            "notes": "Generated via library-oriented heuristic planning with layered modules.",
        }

    @staticmethod
    def _simple_grouping(segments: List[Dict[str, Any]], semantic_keywords: Optional[List[str]] = None) -> List[List[str]]:
        """Fallback simple grouping when dependency analysis doesn't find relationships."""
        semantic_keywords = [k.strip().lower() for k in (semantic_keywords or []) if k.strip()]
        functional_groups = {
            "data": [],
            "config": [],
            "main": [],
            "utils": [],
            "domain": [],
            "integration": [],
            "other": []
        }

        for segment in segments:
            name_lower = segment["name"].lower()
            seg_id = segment["id"]
            if any(kw in name_lower for kw in semantic_keywords):
                functional_groups["domain"].append(seg_id)
                continue

            bucket = LLMPlanner._semantic_bucket(name_lower)
            if bucket in functional_groups:
                functional_groups[bucket].append(seg_id)
            else:
                functional_groups["other"].append(seg_id)

        return [group for group in functional_groups.values() if group]

    @staticmethod
    def _normalize_groups(
        groups: List[List[str]],
        segments: List[Dict[str, Any]],
        target_modules: int,
        min_segments_per_module: int,
        semantic_keywords: Optional[List[str]] = None,
    ) -> List[List[str]]:
        """Predictively normalize group count and avoid tiny modules."""
        if not groups:
            return groups

        semantic_keywords = [k.strip().lower() for k in (semantic_keywords or []) if k.strip()]
        id_to_segment = {s["id"]: s for s in segments}

        def group_signature(group_ids: List[str]) -> Set[str]:
            tokens: Set[str] = set()
            for gid in group_ids:
                seg = id_to_segment.get(gid)
                if not seg:
                    continue
                tokens.update(seg.get("tokens", set()))
                bucket = LLMPlanner._semantic_bucket(seg["name"])
                if bucket != "other":
                    tokens.add(bucket)
            tokens.update(semantic_keywords)
            return tokens

        def similarity(a: List[str], b: List[str]) -> int:
            sa = group_signature(a)
            sb = group_signature(b)
            shared = sa.intersection(sb)
            return len(shared)

        normalized = [list(dict.fromkeys(g)) for g in groups]

        # Merge tiny groups first.
        changed = True
        while changed:
            changed = False
            tiny_idx = next((i for i, g in enumerate(normalized) if len(g) < min_segments_per_module and len(normalized) > 1), None)
            if tiny_idx is None:
                break
            tiny = normalized[tiny_idx]
            best_idx = None
            best_score = -1
            for i, grp in enumerate(normalized):
                if i == tiny_idx:
                    continue
                score = similarity(tiny, grp)
                if score > best_score:
                    best_score = score
                    best_idx = i
            if best_idx is not None:
                normalized[best_idx] = list(dict.fromkeys(normalized[best_idx] + tiny))
                normalized.pop(tiny_idx)
                changed = True

        # Enforce target module count.
        while len(normalized) > target_modules and len(normalized) > 1:
            best_pair = None
            best_score = -1
            for i in range(len(normalized)):
                for j in range(i + 1, len(normalized)):
                    score = similarity(normalized[i], normalized[j])
                    if score > best_score:
                        best_score = score
                        best_pair = (i, j)
            if best_pair is None:
                break
            i, j = best_pair
            normalized[i] = list(dict.fromkeys(normalized[i] + normalized[j]))
            normalized.pop(j)

        return normalized


class ModuleWriter:
    def __init__(self, add_banner: bool = True) -> None:
        self.add_banner = add_banner

    @staticmethod
    def _strip_future_imports(code: str) -> Tuple[str, List[str]]:
        """Remove top-level ``from __future__ import ...`` lines; return (body, future lines)."""
        lines = code.splitlines(keepends=True)
        kept: List[str] = []
        futures: List[str] = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("from __future__ import"):
                futures.append(line.strip())
            else:
                kept.append(line)
        return "".join(kept), futures

    @staticmethod
    def _merge_future_imports(chunks: List[str]) -> List[str]:
        seen: Set[str] = set()
        out: List[str] = []
        for ch in chunks:
            if ch not in seen:
                seen.add(ch)
                out.append(ch)
        return sorted(out)

    @staticmethod
    def _extract_future_imports(source_code: str) -> List[str]:
        return [
            line.strip()
            for line in source_code.splitlines()
            if line.lstrip().startswith("from __future__ import")
        ]

    def write(
        self,
        plan: Dict[str, Any],
        segments: Sequence[Segment],
        output_dir: Path,
        original_name: str,
        source_code: str,
        strict_validation: bool = False,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        seg_map = {seg.identifier: seg for seg in segments}
        architecture_profile = self._detect_architecture_profile_from_segments(segments)
        if architecture_profile == "application_runtime":
            plan = self._promote_runtime_architecture(plan, segments)
        elif architecture_profile == "tool_cli":
            plan = self._promote_tool_architecture(plan, segments)
        plan = self._merge_cyclic_modules(plan, segments)
        manifest_path = output_dir / "module_plan.json"
        previous_written_files: Set[str] = set()
        written_files: List[str] = []

        for cycle_fix_pass in range(3):
            written_files = self._write_modules_once(
                plan=plan,
                segments=segments,
                output_dir=output_dir,
                original_name=original_name,
                source_code=source_code,
                previous_written_files=previous_written_files,
            )
            previous_written_files = set(written_files)

            import_cycles = self._detect_generated_import_cycles(output_dir, written_files)
            if not import_cycles:
                break

            typer.echo(
                "Detected generated import cycles; merging affected modules and rewriting package.",
                err=True,
            )
            plan = self._merge_modules_by_generated_cycles(plan, import_cycles)

        manifest_payload = {
            "original_file": original_name,
            "modules": plan.get("modules", []),
            "shared_helpers": plan.get("shared_helpers"),
            "notes": plan.get("notes"),
            "written_files": written_files,
        }
        manifest_path.write_text(json.dumps(manifest_payload, indent=2))
        init_path = output_dir / "__init__.py"
        init_path.write_text("# Generated by modularizer\n")
        
        # VALIDATE that modules can be executed
        validation = self._validate_modules(output_dir, written_files)
        if strict_validation and not validation["all_valid"]:
            raise RuntimeError(
                f"Generated modules failed validation ({validation['passed']}/{validation['total']} passed)."
            )
        
        return manifest_path

    @staticmethod
    def _detect_architecture_profile_from_segments(segments: Sequence[Segment]) -> str:
        metadata = [
            {
                "kind": seg.kind,
                "name": seg.name,
                "defined_symbols": seg.defined_symbols,
                "signature_excerpt": seg.signature[:200],
            }
            for seg in segments
        ]
        return LLMPlanner._detect_architecture_profile(metadata)

    @staticmethod
    def _promote_tool_architecture(
        plan: Dict[str, Any],
        segments: Sequence[Segment],
    ) -> Dict[str, Any]:
        seg_map = {seg.identifier: seg for seg in segments}
        buckets: Dict[str, List[str]] = {
            "models_types": [],
            "analysis_module": [],
            "llmplanner_module": [],
            "modulewriter_module": [],
            "cli_module": [],
            "shared_module": [],
        }

        for seg in segments:
            bucket = LLMPlanner._tool_cli_bucket(
                name=seg.name,
                kind=seg.kind,
                signature_excerpt=seg.signature[:200],
            )
            if bucket == "validation_module":
                bucket = "modulewriter_module"
            buckets.setdefault(bucket, []).append(seg.identifier)

        modules: List[Dict[str, Any]] = []
        for name, seg_ids in buckets.items():
            if not seg_ids:
                continue
            modules.append(
                {
                    "name": name,
                    "description": f"Tool architecture module for {name.replace('_', ' ')}.",
                    "segment_ids": sorted(seg_ids, key=lambda seg_id: seg_map[seg_id].start_line),
                }
            )

        new_plan = dict(plan)
        new_plan["modules"] = modules
        notes = str(plan.get("notes", "")).strip()
        extra = "Promoted tool architecture: layered modules for models, analysis, planning, writing, CLI, and shared code."
        new_plan["notes"] = f"{notes}\n{extra}".strip() if notes else extra
        return new_plan

    @staticmethod
    def _promote_runtime_architecture(
        plan: Dict[str, Any],
        segments: Sequence[Segment],
    ) -> Dict[str, Any]:
        modules = [m for m in plan.get("modules", []) if isinstance(m, dict)]
        if not modules:
            return plan

        seg_map = {seg.identifier: seg for seg in segments}
        symbol_to_segment: Dict[str, str] = {}
        symbol_usage: Dict[str, int] = {}
        for seg in segments:
            for sym in seg.defined_symbols:
                symbol_to_segment.setdefault(sym, seg.identifier)
            for dep in seg.dependencies:
                symbol_usage[dep] = symbol_usage.get(dep, 0) + 1

        def is_startup_segment(seg: Segment) -> bool:
            code = seg.code
            return (
                "bot.run(" in code
                or 'TOKEN not found in environment variables' in code
                or "Starting Auraxis Bot" in code
                or "print(\"=\" * 50)" in code
            )

        def is_runtime_seed(seg: Segment) -> bool:
            lowered = seg.name.lower()
            code = seg.code.lower()
            if lowered in {"on_ready", "on_message", "get_prefix", "get_guild_prefix", "channel_command_gate"}:
                return True
            if any(
                token in lowered
                for token in [
                    "load_", "save_", "json", "config", "data", "prefix", "channel",
                    "guild", "normalize", "duplicate", "hindi", "rank", "apply_aura",
                    "log_aura_change", "calculate_ai_aura", "evaluate_toxic", "bot",
                ]
            ):
                return True
            if any(
                token in code
                for token in [
                    "commands.bot(",
                    "discord.intents",
                    "os.getenv(\"token\")",
                    "os.getenv('token')",
                    "openrouter_api_key",
                    "load_json_safe(",
                    "config_data =",
                    "aura_data =",
                    "global_data =",
                    "recent_messages =",
                ]
            ):
                return True
            if max((symbol_usage.get(sym, 0) for sym in seg.defined_symbols), default=0) >= 4:
                return True
            return False

        startup_ids: Set[str] = {seg.identifier for seg in segments if is_startup_segment(seg)}
        runtime_ids: Set[str] = {seg.identifier for seg in segments if is_runtime_seed(seg)}

        changed = True
        while changed:
            changed = False
            for seg_id in list(runtime_ids):
                seg = seg_map.get(seg_id)
                if not seg:
                    continue
                for dep in seg.dependencies:
                    target_id = symbol_to_segment.get(dep)
                    if target_id and target_id not in runtime_ids and target_id not in startup_ids:
                        runtime_ids.add(target_id)
                        changed = True

        runtime_ids.difference_update(startup_ids)
        if not runtime_ids:
            return plan

        existing_modules: List[Dict[str, Any]] = []
        used_names: Set[str] = set()
        for module in modules:
            cleaned_segment_ids = [
                seg_id for seg_id in module.get("segment_ids", [])
                if isinstance(seg_id, str)
                and seg_id not in runtime_ids
                and seg_id not in startup_ids
            ]
            if cleaned_segment_ids:
                updated = dict(module)
                updated["segment_ids"] = cleaned_segment_ids
                existing_modules.append(updated)
                used_names.add(str(updated.get("name", "")))

        runtime_name = "runtime_core"
        main_name = "main"
        while runtime_name in used_names:
            runtime_name += "_1"
        used_names.add(runtime_name)
        while main_name in used_names:
            main_name += "_1"

        existing_modules.append(
            {
                "name": runtime_name,
                "description": "Runtime spine: bot object, shared state, bootstrap dependencies, and high-coupling core logic.",
                "segment_ids": sorted(runtime_ids, key=lambda seg_id: seg_map[seg_id].start_line),
            }
        )
        if startup_ids:
            existing_modules.append(
                {
                    "name": main_name,
                    "description": "Dedicated entrypoint module for startup checks and bot.run(...).",
                    "segment_ids": sorted(startup_ids, key=lambda seg_id: seg_map[seg_id].start_line),
                }
            )

        new_plan = dict(plan)
        new_plan["modules"] = existing_modules
        notes = str(plan.get("notes", "")).strip()
        extra = "Promoted runtime architecture: isolated runtime_core and main entrypoint before writing modules."
        new_plan["notes"] = f"{notes}\n{extra}".strip() if notes else extra
        return new_plan

    def _write_modules_once(
        self,
        plan: Dict[str, Any],
        segments: Sequence[Segment],
        output_dir: Path,
        original_name: str,
        source_code: str,
        previous_written_files: Set[str],
    ) -> List[str]:
        seg_map = {seg.identifier: seg for seg in segments}
        written_files: List[str] = []

        all_segment_ids = []
        for module in plan.get("modules", []):
            all_segment_ids.extend(module.get("segment_ids", []))
        if len(all_segment_ids) != len(set(all_segment_ids)):
            typer.echo("Warning: Duplicate segments detected in plan!", err=True)

        all_imports = self._extract_imports(source_code)
        names_from_imports = self._names_bound_by_imports(all_imports)
        dependency_errors = self._check_dependency_coverage(plan, segments, names_from_imports)
        for err in dependency_errors:
            typer.echo(f"Validation warning: {err}", err=True)

        symbol_to_slug: Dict[str, str] = {}
        ambiguous_symbols: Set[str] = set()
        for module in plan.get("modules", []):
            safe = self._slugify(module["name"])
            for seg_id in module.get("segment_ids", []):
                seg = seg_map.get(seg_id)
                if not seg:
                    continue
                for sym in seg.defined_symbols:
                    if sym in symbol_to_slug and symbol_to_slug[sym] != safe:
                        ambiguous_symbols.add(sym)
                    else:
                        symbol_to_slug[sym] = safe
        for sym in ambiguous_symbols:
            symbol_to_slug.pop(sym, None)

        for module in plan.get("modules", []):
            raw_name = module["name"]
            safe_name = self._slugify(raw_name)
            target = output_dir / f"{safe_name}.py"
            written_files.append(target.name)

            module_code = ""
            module_dependencies = set()
            module_used_attributes: List[Tuple[str, str]] = []
            written_segment_ids = set()
            locally_defined: Set[str] = set()
            future_chunks: List[str] = []

            for seg_id in module.get("segment_ids", []):
                if seg_id in written_segment_ids:
                    typer.echo(f"Warning: Segment {seg_id} already written to module {raw_name}", err=True)
                    continue
                written_segment_ids.add(seg_id)

                seg = seg_map.get(seg_id)
                if not seg:
                    typer.echo(f"Warning: missing segment {seg_id}", err=True)
                    continue
                locally_defined.update(seg.defined_symbols)
                body, fut_lines = self._strip_future_imports(seg.code)
                future_chunks.extend(fut_lines)
                module_code += f"# --- segment {seg.identifier} ({seg.kind}) ---\n{body}\n\n"
                module_dependencies.update(seg.dependencies)
                module_used_attributes.extend(seg.used_attributes)

            global_future_lines = self._extract_future_imports(source_code)
            future_lines = self._merge_future_imports(future_chunks + global_future_lines)
            needed_imports = self._get_needed_imports(
                all_imports,
                module_code,
                module_dependencies,
                safe_name,
                used_attributes=module_used_attributes,
                symbol_to_slug=symbol_to_slug,
                locally_defined=locally_defined,
            )

            with target.open("w", encoding="utf-8") as handle:
                if self.add_banner:
                    handle.write(
                        textwrap.dedent(
                            f'''"""
                            Auto-generated module from {original_name}
                            Plan summary: {module.get("description","(no description)")}
                            """
                            '''
                        ).strip()
                        + "\n\n"
                    )
                if future_lines:
                    handle.write("\n".join(future_lines) + "\n\n")
                if needed_imports:
                    handle.write("\n".join(needed_imports) + "\n\n")
                handle.write(module_code)

        obsolete = previous_written_files - set(written_files)
        for filename in obsolete:
            target = output_dir / filename
            if target.exists():
                try:
                    target.unlink()
                except OSError:
                    pass

        return written_files

    @staticmethod
    def _names_bound_by_imports(all_imports: List[Tuple[str, List[str]]]) -> Set[str]:
        return {n for _, names in all_imports for n in names}

    @staticmethod
    def _check_dependency_coverage(
        plan: Dict[str, Any],
        segments: Sequence[Segment],
        names_from_imports: Set[str],
    ) -> List[str]:
        """Ensure cross-module dependencies point to symbols in some generated module."""
        errors: List[str] = []
        seg_map = {seg.identifier: seg for seg in segments}
        symbol_to_module: Dict[str, str] = {}
        for module in plan.get("modules", []):
            module_name = module.get("name", "unknown_module")
            for seg_id in module.get("segment_ids", []):
                seg = seg_map.get(seg_id)
                if seg:
                    for sym in seg.defined_symbols:
                        symbol_to_module[sym] = module_name

        for module in plan.get("modules", []):
            module_name = module.get("name", "unknown_module")
            local_symbols: Set[str] = set()
            external_deps: Set[str] = set()
            for seg_id in module.get("segment_ids", []):
                seg = seg_map.get(seg_id)
                if not seg:
                    continue
                local_symbols.update(seg.defined_symbols)
                for dep in seg.dependencies:
                    if dep not in local_symbols:
                        external_deps.add(dep)

            for dep in sorted(external_deps):
                if dep in names_from_imports:
                    continue
                target_module = symbol_to_module.get(dep)
                if target_module is None:
                    errors.append(f"Module '{module_name}' depends on '{dep}', but no module defines it.")
        return errors

    @staticmethod
    def _extract_imports(source_code: str) -> List[Tuple[str, List[str]]]:
        """
        Extract imports using AST (not string parsing).
        Returns list of (import_statement, imported_names).
        """
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []
        
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    import_stmt = f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else "")
                    imports.append((import_stmt, [name]))
            
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                level_dots = "." * node.level
                from_part = f"from {level_dots}{module_name} import"
                
                imported_names = []
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    imported_names.append(name)
                
                def _render_alias(a: ast.alias) -> str:
                    return f"{a.name} as {a.asname}" if a.asname else a.name

                import_stmt = f"{from_part} {', '.join(_render_alias(a) for a in node.names)}"
                imports.append((import_stmt, imported_names))
        
        return imports

    def _get_needed_imports(
        self, 
        all_imports: List[Tuple[str, List[str]]], 
        module_code: str, 
        dependencies: set,
        current_module_name: str,
        used_attributes: List[Tuple[str, str]] = None,
        symbol_to_slug: Optional[Dict[str, str]] = None,
        locally_defined: Optional[Set[str]] = None,
    ) -> List[str]:
        """
        Determine which imports this module needs.
        Enhanced with context-aware detection, attribute filtering, symbol types.
        
        Args:
            all_imports: List of (import_statement, imported_names) tuples
            module_code: The code in this module
            dependencies: Set of segment names this module depends on
            current_module_name: Slugified stem of the current output module
            used_attributes: List of (object, attribute) tuples to exclude attributes
        """
        if used_attributes is None:
            used_attributes = []
        if symbol_to_slug is None:
            symbol_to_slug = {}
        if locally_defined is None:
            locally_defined = set()
        
        attribute_names = {attr for obj, attr in used_attributes}
        needed_imports = []
        
        cross_module_deps = {
            dep_name
            for dep_name in dependencies
            if dep_name not in locally_defined
            and dep_name in symbol_to_slug
            and symbol_to_slug.get(dep_name) != current_module_name
        }

        def _filter_import_stmt(import_stmt: str, keep_names: List[str]) -> Optional[str]:
            if not keep_names:
                return None
            if import_stmt.startswith("import "):
                parts = [part.strip() for part in import_stmt[7:].split(",")]
                kept = []
                for part in parts:
                    alias = part.split(" as ")[-1].strip()
                    base = part.split(" as ")[0].strip()
                    if alias in keep_names or base in keep_names:
                        kept.append(part)
                return "import " + ", ".join(kept) if kept else None
            if import_stmt.startswith("from ") and " import " in import_stmt:
                prefix, imported = import_stmt.rsplit(" import ", 1)
                parts = [part.strip() for part in imported.split(",")]
                kept = []
                for part in parts:
                    alias = part.split(" as ")[-1].strip()
                    base = part.split(" as ")[0].strip()
                    if alias in keep_names or base in keep_names:
                        kept.append(part)
                return f"{prefix} import {', '.join(kept)}" if kept else None
            return import_stmt

        for import_stmt, imported_names in all_imports:
            # Filter: don't import names that are attributes
            valid_names = [n for n in imported_names if n not in attribute_names]
            if not valid_names:
                continue

            keep_names = [n for n in valid_names if n not in cross_module_deps]
            import_needed = False

            for name in keep_names:
                pattern = r'\b' + re.escape(name) + r'\b'
                if re.search(pattern, module_code) and not self._is_in_comment(module_code, name):
                    import_needed = True
                    break
                if name in dependencies:
                    import_needed = True
                    break

            if import_needed:
                filtered = _filter_import_stmt(import_stmt, keep_names)
                if filtered:
                    needed_imports.append(filtered)

        # Add cross-module imports for generated symbols only.
        for dep_name in sorted(dependencies):
            if dep_name in attribute_names:
                continue
            if dep_name in locally_defined:
                continue
            target_slug = symbol_to_slug.get(dep_name)
            if not target_slug or target_slug == current_module_name:
                continue
            cross_import = f"from .{target_slug} import {dep_name}"
            if cross_import not in needed_imports:
                needed_imports.append(cross_import)

        # FINAL FILTER: Remove any imports that import locally-defined symbols
        # This prevents circular imports within the same module
        final_imports = []
        for import_stmt in needed_imports:
            # Parse import statement to extract imported names
            imported_names = set()
            if import_stmt.startswith("from ") and " import " in import_stmt:
                _, right = import_stmt.rsplit(" import ", 1)
                for item in right.split(","):
                    item = item.strip()
                    # Handle "name as alias"
                    if " as " in item:
                        imported_names.add(item.split(" as ")[-1].strip())
                    else:
                        imported_names.add(item)
            
            # Only keep imports that don't import locally-defined symbols
            if not imported_names.intersection(locally_defined):
                final_imports.append(import_stmt)
        
        return final_imports
    
    @staticmethod
    def _is_in_comment(code: str, name: str) -> bool:
        """Check if a name appears only in comments (simple check)."""
        lines = code.split('\n')
        for line in lines:
            if '#' in line:
                code_part = line.split('#', 1)[0]
            else:
                code_part = line
            
            pattern = r'\b' + re.escape(name) + r'\b'
            if re.search(pattern, code_part):
                return False
        
        return True

    @staticmethod
    def _parse_import_statement(import_stmt: str) -> List[str]:
        """Parse an import statement to extract the names it makes available."""
        names = []
        import_stmt = import_stmt.strip()
        
        if import_stmt.startswith('import '):
            parts = import_stmt[7:].split(',')
            for part in parts:
                name = part.split(' as ')[0].strip()
                names.append(name.split('.')[-1])
                
        elif import_stmt.startswith('from '):
            if ' import ' in import_stmt:
                module_part, imports_part = import_stmt.split(' import ', 1)
                import_items = imports_part.split(',')
                for item in import_items:
                    name = item.split(' as ')[0].strip()
                    if name != '*':
                        names.append(name)
        
        return names

    @staticmethod
    def _detect_generated_import_cycles(output_dir: Path, written_files: List[str]) -> List[List[str]]:
        graph = ModuleWriter._build_generated_import_graph(output_dir, written_files)
        sccs = ModuleWriter._strongly_connected_components(graph)
        return [sorted(component) for component in sccs if len(component) > 1]

    @staticmethod
    def _build_generated_import_graph(output_dir: Path, written_files: List[str]) -> Dict[str, Set[str]]:
        graph: Dict[str, Set[str]] = {}
        module_names = {Path(filename).stem for filename in written_files if filename.endswith(".py")}

        for filename in written_files:
            if not filename.endswith(".py"):
                continue
            module_name = Path(filename).stem
            graph.setdefault(module_name, set())
            filepath = output_dir / filename
            try:
                tree = ast.parse(filepath.read_text(encoding="utf-8"))
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if node.level <= 0 or not node.module:
                    continue
                imported_module = node.module.split(".", 1)[0]
                if imported_module in module_names and imported_module != module_name:
                    graph[module_name].add(imported_module)

        return graph

    def _merge_modules_by_generated_cycles(
        self,
        plan: Dict[str, Any],
        import_cycles: List[List[str]],
    ) -> Dict[str, Any]:
        modules = [m for m in plan.get("modules", []) if isinstance(m, dict)]
        if not modules or not import_cycles:
            return plan

        slug_to_module = {self._slugify(str(module.get("name", ""))): module for module in modules}
        merged_modules: List[Dict[str, Any]] = []
        merged_slugs: Set[str] = set()

        for cycle in import_cycles:
            cycle_modules = [slug_to_module[slug] for slug in cycle if slug in slug_to_module]
            if len(cycle_modules) < 2:
                continue

            merged_slugs.update(cycle)
            descriptions: List[str] = []
            merged_segment_ids: List[str] = []
            seen_segments: Set[str] = set()
            merged_name_parts: List[str] = []

            for module in cycle_modules:
                module_name = str(module.get("name", "")).strip()
                if module_name:
                    merged_name_parts.append(self._slugify(module_name))
                description = str(module.get("description", "")).strip()
                if description:
                    descriptions.append(description)
                for seg_id in module.get("segment_ids", []):
                    if isinstance(seg_id, str) and seg_id not in seen_segments:
                        seen_segments.add(seg_id)
                        merged_segment_ids.append(seg_id)

            merged_name = "_".join(dict.fromkeys(merged_name_parts)) or "merged_cycle_module"
            merged_modules.append(
                {
                    "name": merged_name,
                    "description": " / ".join(dict.fromkeys(descriptions))
                    or f"Merged generated import cycle: {', '.join(cycle)}",
                    "segment_ids": merged_segment_ids,
                }
            )

        final_modules: List[Dict[str, Any]] = []
        for module in modules:
            slug = self._slugify(str(module.get("name", "")))
            if slug not in merged_slugs:
                final_modules.append(module)
        final_modules.extend(merged_modules)

        merged_plan = dict(plan)
        merged_plan["modules"] = final_modules
        merged_notes = str(plan.get("notes", "")).strip()
        cycle_note = "Auto-merged modules to resolve generated import cycles."
        merged_plan["notes"] = f"{merged_notes}\n{cycle_note}".strip()
        return merged_plan

    @staticmethod
    def _sort_modules_for_validation(output_dir: Path, written_files: List[str]) -> List[str]:
        """Order modules so relative imports load dependencies first (best-effort; cycles fall back)."""
        stems = {Path(f).stem for f in written_files}
        if len(stems) <= 1:
            return list(written_files)

        graph: Dict[str, Set[str]] = {Path(f).stem: set() for f in written_files}
        indegree: Dict[str, int] = {Path(f).stem: 0 for f in written_files}

        rel_imp = re.compile(r"^\s*from \.([a-zA-Z_][a-zA-Z0-9_]*) import ", re.MULTILINE)
        for filename in written_files:
            stem = Path(filename).stem
            path = output_dir / filename
            try:
                code = path.read_text(encoding="utf-8")
            except OSError:
                continue
            for m in rel_imp.finditer(code):
                dep = m.group(1)
                if dep not in stems or dep == stem:
                    continue
                # dep must load before stem -> edge dep -> stem
                if stem not in graph[dep]:
                    graph[dep].add(stem)
                    indegree[stem] = indegree.get(stem, 0) + 1

        queue = [s for s in indegree if indegree[s] == 0]
        queue.sort()
        order: List[str] = []
        while queue:
            n = queue.pop(0)
            order.append(n)
            for nbr in sorted(graph.get(n, ())):
                indegree[nbr] -= 1
                if indegree[nbr] == 0:
                    queue.append(nbr)
                    queue.sort()

        if len(order) != len(stems):
            return list(written_files)

        stem_to_file = {Path(f).stem: f for f in written_files}
        return [stem_to_file[s] for s in order if s in stem_to_file]

    @staticmethod
    def _strongly_connected_components(graph: Dict[str, Set[str]]) -> List[List[str]]:
        index: Dict[str, int] = {}
        lowlink: Dict[str, int] = {}
        stack: List[str] = []
        on_stack: Set[str] = set()
        result: List[List[str]] = []

        def strongconnect(node: str) -> None:
            index[node] = len(index)
            lowlink[node] = index[node]
            stack.append(node)
            on_stack.add(node)

            for neighbor in graph.get(node, set()):
                if neighbor not in index:
                    strongconnect(neighbor)
                    lowlink[node] = min(lowlink[node], lowlink[neighbor])
                elif neighbor in on_stack:
                    lowlink[node] = min(lowlink[node], index[neighbor])

            if lowlink[node] == index[node]:
                component: List[str] = []
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    component.append(w)
                    if w == node:
                        break
                result.append(component)

        for node in graph:
            if node not in index:
                strongconnect(node)

        return result

    @staticmethod
    def _merge_cyclic_modules(plan: Dict[str, Any], segments: Sequence[Segment]) -> Dict[str, Any]:
        modules = [m for m in plan.get("modules", []) if isinstance(m, dict)]
        if not modules:
            return plan

        metadata = [
            {"segment_id": seg.identifier, "dependencies": seg.dependencies}
            for seg in segments
        ]
        module_deps = LLMPlanner._build_module_dependencies({"modules": modules}, metadata)
        sccs = ModuleWriter._strongly_connected_components(module_deps)
        cyclic_groups = [sorted(comp) for comp in sccs if len(comp) > 1]
        if not cyclic_groups:
            return plan

        module_map = {module["name"]: module for module in modules}
        merged_modules: List[Dict[str, Any]] = []
        merged_names: Set[str] = set()
        total_segments = sum(
            len([seg_id for seg_id in module.get("segment_ids", []) if isinstance(seg_id, str)])
            for module in modules
        )
        skipped_cycles: List[List[str]] = []

        for cycle in cyclic_groups:
            merged_segment_ids: List[str] = []
            seen_segments: Set[str] = set()
            descriptions: List[str] = []
            for module_name in cycle:
                module = module_map.get(module_name)
                if not module:
                    continue
                if desc := str(module.get("description", "")).strip():
                    descriptions.append(desc)
                for seg_id in module.get("segment_ids", []):
                    if seg_id not in seen_segments:
                        merged_segment_ids.append(seg_id)
                        seen_segments.add(seg_id)

            # Keep feature-oriented structure intact: only auto-merge small/local cycles.
            if len(cycle) > 3:
                skipped_cycles.append(cycle)
                continue
            if total_segments and len(merged_segment_ids) / total_segments > 0.35:
                skipped_cycles.append(cycle)
                continue

            merged_names.update(cycle)
            merged_name = "_".join(cycle)
            merged_description = (
                " / ".join(dict.fromkeys(descriptions))
                or f"Merged cyclic modules: {', '.join(cycle)}"
            )
            merged_modules.append(
                {
                    "name": merged_name,
                    "description": merged_description,
                    "segment_ids": merged_segment_ids,
                }
            )

        final_modules: List[Dict[str, Any]] = []
        for module in modules:
            if module["name"] not in merged_names:
                final_modules.append(module)
        final_modules.extend(merged_modules)

        merged_plan = dict(plan)
        merged_plan["modules"] = final_modules
        merged_notes = str(plan.get("notes", "")).strip()
        note_parts: List[str] = []
        merged_flat = sorted([n for group in cyclic_groups if group not in skipped_cycles for n in group])
        skipped_flat = sorted([n for group in skipped_cycles for n in group])
        if merged_flat:
            note_parts.append(f"Auto-merged cyclic modules: {', '.join(merged_flat)}.")
        if skipped_flat:
            note_parts.append(
                f"Preserved larger cyclic module groups to avoid collapsing the plan: {', '.join(skipped_flat)}."
            )
        if merged_notes:
            note_parts.insert(0, merged_notes)
        merged_plan["notes"] = "\n".join(note_parts)
        return merged_plan

    @staticmethod
    def _validate_modules(output_dir: Path, written_files: List[str]) -> Dict[str, Any]:
        """
        Validate generated modules without executing their runtime dependencies.
        Checks:
        - Syntax validation
        - Bytecode compilation
        - Package-local relative import target resolution
        - Optional import execution when explicitly enabled via env var
        Reports ✅/❌ status for each module
        """
        typer.echo("Validating generated modules...", err=True)
        
        all_valid = True
        tested_modules: Dict[str, bool] = {}
        package_name = output_dir.name
        package_parent = str(output_dir.parent.resolve())
        enable_import_validation = os.environ.get("MODULIZER_IMPORT_VALIDATE", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        inserted_path = False
        if package_parent not in sys.path:
            sys.path.insert(0, package_parent)
            inserted_path = True
        
        try:
            ordered_files = ModuleWriter._sort_modules_for_validation(output_dir, written_files)
            for filename in ordered_files:
                filepath = output_dir / filename
                module_name = filepath.stem
                fq_module_name = f"{package_name}.{module_name}"
                status = "⏳"
                error_msg = ""

                try:
                    # 1. Syntax check
                    code = filepath.read_text(encoding="utf-8")
                    ast.parse(code)

                    # 2. Compile without executing module top-level code.
                    compile(code, str(filepath), "exec")

                    # 3. Validate package-local relative imports without importing third-party deps.
                    ModuleWriter._validate_relative_import_targets(
                        code=code,
                        output_dir=output_dir,
                        package_name=package_name,
                    )

                    # 4. Optional deep validation for environments that want execution checks.
                    if enable_import_validation:
                        sys.modules.pop(fq_module_name, None)
                        imported = importlib.import_module(fq_module_name)
                        importlib.reload(imported)
                    status = "✅"
                    tested_modules[module_name] = True
                except SyntaxError as e:
                    status = "❌"
                    error_msg = f"Syntax error: {e.msg} at line {e.lineno}"
                    tested_modules[module_name] = False
                    all_valid = False
                except Exception as e:
                    status = "❌"
                    error_msg = f"{type(e).__name__}: {e}"
                    tested_modules[module_name] = False
                    all_valid = False

                if error_msg:
                    typer.echo(f"{status} {filename} - {error_msg[:140]}", err=True)
                else:
                    typer.echo(f"{status} {filename} is valid and executable", err=True)

            import_cycles = ModuleWriter._detect_generated_import_cycles(output_dir, written_files)
            if import_cycles:
                all_valid = False
                cycle_info = ", ".join(" -> ".join(cycle + [cycle[0]]) for cycle in import_cycles)
                typer.echo(f"❌ Generated import cycles remain: {cycle_info}", err=True)
        finally:
            if inserted_path:
                try:
                    sys.path.remove(package_parent)
                except ValueError:
                    pass
        
        # Summary
        passed = sum(1 for v in tested_modules.values() if v)
        total = len(tested_modules)
        typer.echo(f"\nValidation summary: {passed}/{total} modules passed", err=True)
        
        if not all_valid:
            typer.echo("WARNING: Module validation revealed issues. Review errors above.", err=True)
        return {"all_valid": all_valid, "passed": passed, "total": total}

    @staticmethod
    def _validate_relative_import_targets(code: str, output_dir: Path, package_name: str) -> None:
        """Ensure intra-package relative imports point at generated modules or package init."""
        tree = ast.parse(code)
        known_modules = {path.stem for path in output_dir.glob("*.py")}
        known_modules.add(package_name)

        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level <= 0:
                continue

            target_module = node.module
            if not target_module:
                continue

            root_name = target_module.split(".", 1)[0]
            if root_name not in known_modules:
                raise ImportError(
                    f"Relative import target '.{target_module}' not found in generated package"
                )

    @staticmethod
    def _slugify(value: str) -> str:
        value = value.lower().strip()
        value = re.sub(r"[^a-z0-9]+", "_", value)
        value = re.sub(r"_+", "_", value).strip("_")
        return value or "module"


@app.command()
def version() -> None:
    """Show version information."""
    typer.echo("modulizer 0.1.0")


@app.command()
def init_config(output_file: Path = typer.Option("modulizer_config.json", help="Path to create config file.")) -> None:
    """Generate a sample configuration file."""
    sample_config = {
        "model": LLMPlanner.DEFAULT_MODEL,
        "api_key": "<YOUR_API_KEY>",
        "openai_base_url": LLMPlanner.DEFAULT_BASE_URL,
        "planning_mode": "safe",
        "temperature": 0.9,
        "top_p": 0.3,
        "top_k": 20,
        "frequency_penalty": 0.8,
        "verbose": False,
        "max_modules": 8,
        "min_segments_per_module": 2,
        "semantic_grouping": True,
        "semantic_keywords": [],
        "ai_retries": 5,
        "strict_validation": False,
        "heuristic_fallback": True,
    }
    try:
        output_file.write_text(json.dumps(sample_config, indent=2))
        typer.echo(f"Sample config written to {output_file}")
    except IOError as e:
        typer.echo(f"Error: Failed to write config file: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def modularize(
    input_file: Path = typer.Option(
        ...,
        exists=True,
        readable=True,
        help="Python file to split.",
    ),
    output_dir: Path = typer.Option(..., help="Directory to write modules."),
    model: Optional[str] = typer.Option(
        None,
        help=f"Chat model (default {LLMPlanner.DEFAULT_MODEL!r}).",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        help="API key (overrides OPENAI_API_KEY).",
    ),
    offline: bool = typer.Option(False, help="Skip AI entirely; use heuristic plan only."),
    openai_base_url: Optional[str] = typer.Option(
        None,
        help=f"API base URL (overrides OPENAI_BASE_URL; default {LLMPlanner.DEFAULT_BASE_URL!r}).",
    ),
    planning_mode: Optional[str] = typer.Option(
        None,
        help="Planning mode: safe, hybrid, or ai_first.",
    ),
    temperature: Optional[float] = typer.Option(
        None,
        min=0.0,
        max=2.0,
        help="Sampling temperature (default 0.9).",
    ),
    top_p: Optional[float] = typer.Option(
        None,
        min=0.0,
        max=1.0,
        help="Nucleus sampling top_p (default 0.3).",
    ),
    top_k: Optional[int] = typer.Option(
        None,
        min=1,
        help="Top-k sampling (default 20).",
    ),
    frequency_penalty: Optional[float] = typer.Option(
        None,
        min=-2.0,
        max=2.0,
        help="Frequency penalty (default 0.8).",
    ),
    config: Optional[Path] = typer.Option(
        None,
        help="JSON configuration file to load default settings from.",
    ),
    max_modules: Optional[int] = typer.Option(
        None,
        min=1,
        max=64,
        help="Target maximum number of output modules.",
    ),
    min_segments_per_module: Optional[int] = typer.Option(
        None,
        min=1,
        max=20,
        help="Minimum segments per module target (planner tries to avoid tiny modules).",
    ),
    semantic_grouping: Optional[bool] = typer.Option(
        None,
        "--semantic-grouping/--no-semantic-grouping",
        help="Use semantic naming signals to keep related symbols together.",
    ),
    semantic_keywords: Optional[str] = typer.Option(
        None,
        help="Comma-separated domain keywords to bias semantic grouping.",
    ),
    strict_validation: Optional[bool] = typer.Option(
        None,
        "--strict-validation/--no-strict-validation",
        help="Fail the run if dependency coverage or module validation reports issues.",
    ),
    ai_retries: Optional[int] = typer.Option(
        None,
        min=1,
        max=15,
        help="Maximum AI planning retries (default 5). Heuristic fallback only if --heuristic-fallback.",
    ),
    heuristic_fallback: Optional[bool] = typer.Option(
        None,
        "--heuristic-fallback/--no-heuristic-fallback",
        help="If AI fails after retries, fall back to heuristic plan (default: off — prioritize AI).",
    ),
    verbose: bool = typer.Option(False, help="Enable verbose output."),
) -> None:
    # If called programmatically (e.g. from the GUI), Typer's defaults may be OptionInfo objects.
    # Coerce them away so config/default logic works and AI payloads stay JSON-serializable.
    def _coerce_option_default(value: Any) -> Any:
        return None if isinstance(value, OptionInfo) else value

    model = _coerce_option_default(model)
    api_key = _coerce_option_default(api_key)
    openai_base_url = _coerce_option_default(openai_base_url)
    planning_mode = _coerce_option_default(planning_mode)
    temperature = _coerce_option_default(temperature)
    top_p = _coerce_option_default(top_p)
    top_k = _coerce_option_default(top_k)
    frequency_penalty = _coerce_option_default(frequency_penalty)
    config = _coerce_option_default(config)
    max_modules = _coerce_option_default(max_modules)
    min_segments_per_module = _coerce_option_default(min_segments_per_module)
    semantic_grouping = _coerce_option_default(semantic_grouping)
    semantic_keywords = _coerce_option_default(semantic_keywords)
    strict_validation = _coerce_option_default(strict_validation)
    ai_retries = _coerce_option_default(ai_retries)
    heuristic_fallback = _coerce_option_default(heuristic_fallback)

    # Load configuration file if provided
    config_data = {}
    if config and config.exists():
        try:
            config_data = json.loads(config.read_text())
        except (json.JSONDecodeError, IOError) as e:
            typer.echo(f"Warning: Failed to load config file {config}: {e}", err=True)
    
    # Apply config defaults (command line options take precedence)
    model = model or config_data.get("model") or LLMPlanner.DEFAULT_MODEL
    api_key = api_key or config_data.get("api_key")
    openai_base_url = openai_base_url or config_data.get("openai_base_url")
    planning_mode = str(planning_mode or config_data.get("planning_mode") or "safe").strip().lower()
    temperature = temperature if temperature is not None else float(config_data.get("temperature", 0.9))
    top_p = top_p if top_p is not None else float(config_data.get("top_p", 0.3))
    top_k = top_k if top_k is not None else int(config_data.get("top_k", 20))
    frequency_penalty = (
        frequency_penalty if frequency_penalty is not None else float(config_data.get("frequency_penalty", 0.8))
    )
    verbose = verbose or config_data.get("verbose", False)
    max_modules = max_modules or config_data.get("max_modules", 8)
    min_segments_per_module = min_segments_per_module or config_data.get("min_segments_per_module", 2)
    semantic_grouping = semantic_grouping if semantic_grouping is not None else bool(config_data.get("semantic_grouping", True))
    strict_validation = strict_validation if strict_validation is not None else bool(config_data.get("strict_validation", False))
    ai_retries = ai_retries or config_data.get("ai_retries", 5)
    heuristic_fallback = (
        heuristic_fallback
        if heuristic_fallback is not None
        else bool(config_data.get("heuristic_fallback", True))
    )
    if planning_mode in {"ai-first", "ai first"}:
        planning_mode = "ai_first"
    if planning_mode not in LLMPlanner.PLANNING_MODES:
        typer.echo(
            f"Warning: Unknown planning mode {planning_mode!r}; using 'safe' instead.",
            err=True,
        )
        planning_mode = "safe"
    if planning_mode == "safe":
        offline = True
        heuristic_fallback = True
    elif planning_mode == "hybrid":
        heuristic_fallback = True
    raw_keywords = semantic_keywords if semantic_keywords is not None else config_data.get("semantic_keywords", [])
    if isinstance(raw_keywords, str):
        semantic_keywords_list = [k.strip().lower() for k in raw_keywords.split(",") if k.strip()]
    elif isinstance(raw_keywords, list):
        semantic_keywords_list = [str(k).strip().lower() for k in raw_keywords if str(k).strip()]
    else:
        semantic_keywords_list = []

    # Input validation
    if input_file.suffix != ".py":
        typer.echo("Error: Input file must be a Python file (.py)", err=True)
        raise typer.Exit(1)
    
    if output_dir.exists() and not output_dir.is_dir():
        typer.echo("Error: Output path exists but is not a directory", err=True)
        raise typer.Exit(1)
    
    if output_dir == input_file.parent:
        typer.echo("Warning: Output directory is the same as input file directory. This may overwrite files.", err=True)
        if not typer.confirm("Continue anyway?"):
            raise typer.Exit(0)
    
    try:
        if verbose:
            typer.echo("Analyzing source file...", err=True)
        analyzer = SourceAnalyzer(input_file)
        summary, segments = analyzer.analyze()
        typer.echo(summary)
    except SyntaxError as e:
        typer.echo(f"Error: Invalid Python syntax in {input_file}: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error: Failed to analyze {input_file}: {e}", err=True)
        raise typer.Exit(1)

    if verbose:
        typer.echo(f"Planning module structure with {len(segments)} segments...", err=True)
        typer.echo(f"Planning mode: {planning_mode}", err=True)
    planner = LLMPlanner(
        model=model,
        api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        frequency_penalty=frequency_penalty,
        offline=offline,
        max_retries=ai_retries,
        max_modules=max_modules,
        min_segments_per_module=min_segments_per_module,
        semantic_grouping=semantic_grouping,
        semantic_keywords=semantic_keywords_list,
        base_url=openai_base_url,
        verbose=verbose,
        allow_heuristic_fallback=heuristic_fallback,
        planning_mode=planning_mode,
    )
    try:
        plan = planner.plan(summary, segments)
    except Exception as e:
        typer.echo(f"Error: Failed to generate plan: {e}", err=True)
        typer.echo(
            "Use --heuristic-fallback to allow heuristic planning after AI failure, or --offline for heuristic-only.",
            err=True,
        )
        raise typer.Exit(1)
    
    if verbose:
        typer.echo(f"Writing {len(plan.get('modules', []))} modules to {output_dir}...", err=True)
    writer = ModuleWriter()
    try:
        # Read the source code to extract imports
        source_code = input_file.read_text(encoding="utf-8")
        manifest = writer.write(
            plan,
            segments,
            output_dir,
            input_file.name,
            source_code,
            strict_validation=strict_validation,
        )
        typer.echo(f"Modules written. Manifest: {manifest}")
    except Exception as e:
        typer.echo(f"Error: Failed to write modules: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app() 
