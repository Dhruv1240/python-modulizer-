from __future__ import annotations

import json
import os
import re
import textwrap
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import typer
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError
from typer.models import OptionInfo

from .models import Segment


class LLMPlanner:
    """Matches typical OpenAI-client usage (e.g. AIMLAPI at https://ai.aimlapi.com)."""

    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
    DEFAULT_MODEL = ""
    MODEL_ENV_VARS = ("MODULIZER_MODEL", "OPENAI_MODEL", "OPENROUTER_MODEL", "LLM_MODEL")
    PLANNING_MODES = {"safe", "hybrid", "ai_first"}

    @staticmethod
    def _normalize_option(value: Any, default: Any) -> Any:
        if isinstance(value, OptionInfo):
            return default
        return value

    @classmethod
    def resolve_model(cls, value: Any = None) -> Optional[str]:
        normalized = cls._normalize_option(value, None)
        if isinstance(normalized, str):
            normalized = normalized.strip()
            if normalized:
                return normalized
        for env_name in cls.MODEL_ENV_VARS:
            env_value = os.environ.get(env_name)
            if env_value and env_value.strip():
                return env_value.strip()
        return None

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
        self.model = self.resolve_model(model)
        self.temperature = float(self._normalize_option(temperature, 0.9))
        self.top_p = float(self._normalize_option(top_p, 0.3))
        self.top_k = int(self._normalize_option(top_k, 20))
        self.frequency_penalty = float(self._normalize_option(frequency_penalty, 0.8))
        self.offline = bool(self._normalize_option(offline, False))
        self.max_retries = max(1, int(self._normalize_option(max_retries, 5)))
        self.allow_heuristic_fallback = bool(self._normalize_option(allow_heuristic_fallback, False))
        requested_mode = str(self._normalize_option(planning_mode, "safe") or "safe").strip().lower()
        aliases = {"ai-first": "ai_first", "ai first": "ai_first"}
        requested_mode = aliases.get(requested_mode, requested_mode)
        self.planning_mode = requested_mode if requested_mode in self.PLANNING_MODES else "safe"
        self.max_modules = max(1, int(self._normalize_option(max_modules, 8)))
        self.min_segments_per_module = max(1, int(self._normalize_option(min_segments_per_module, 2)))
        self.semantic_grouping = bool(self._normalize_option(semantic_grouping, True))
        self.semantic_keywords = [
            keyword.strip().lower()
            for keyword in (self._normalize_option(semantic_keywords, []) or [])
            if keyword and str(keyword).strip()
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
            if not self.model:
                raise RuntimeError(
                    'Missing model. Use --model, set "model" in the config file, or set '
                    "MODULIZER_MODEL, OPENAI_MODEL, OPENROUTER_MODEL, or LLM_MODEL."
                )
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key and self.planning_mode == "ai_first":
                raise RuntimeError("Missing API key. Use --api-key or set OPENAI_API_KEY.")
            if key:
                self._openai_client = OpenAI(
                    base_url=raw_base,
                    api_key=key,
                    timeout=120.0,
                )

    @staticmethod
    def _message_text(message: Any) -> Optional[str]:
        if message is None:
            return None
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
                    " ".join(str(sym) for sym in entry.get("defined_symbols", []) if isinstance(sym, str)),
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
                "runtime.core, main, data.storage, features.analytics, features.visuals, "
                "features.battle, features.economy, features.progression, commands.general."
            )
        if profile == "tool_cli":
            return (
                "Architecture target: tool/CLI file. Prefer layered modules such as "
                "core.models, core.analysis, core.planning, core.writing, app.cli, shared.helpers. "
                "Do not force a runtime_core/main bot-style architecture."
            )
        if profile == "generic_library":
            return (
                "Architecture target: reusable library/module file. Prefer modules such as "
                "core.models, core.analysis, core.processing, io.storage, validation.rules, "
                "api.routes, shared.helpers."
            )
        return "Architecture target: generic Python module. Prefer cohesive, low-coupling modules and avoid monolithic output."

    @staticmethod
    def _package_style_module_name(name: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9._/]+", "_", str(name or "").strip()).strip("._/")
        normalized = normalized.replace("/", ".")
        normalized = re.sub(r"\.+", ".", normalized)
        if not normalized:
            return "module"

        direct_map = {
            "models": "core.models",
            "analysis": "core.analysis",
            "planning": "core.planning",
            "writing": "core.writing",
            "processing": "core.processing",
            "cli": "app.cli",
            "shared": "shared.helpers",
            "config": "config.settings",
            "bot_core": "runtime.bot_core",
            "runtime_core": "runtime.core",
            "data_storage": "data.storage",
            "commands_general": "commands.general",
            "commands_admin": "commands.admin",
            "commands_economy": "commands.economy",
            "commands_analytics": "commands.analytics",
            "aura_commands": "commands.aura",
            "analytics": "features.analytics",
            "visuals": "features.visuals",
            "battle": "features.battle",
            "economy": "features.economy",
            "progression": "features.progression",
            "moderation": "features.moderation",
            "io": "io.storage",
            "validation": "validation.rules",
            "api": "api.routes",
        }
        if normalized in direct_map:
            return direct_map[normalized]
        if "." in normalized:
            return normalized
        if "_" in normalized:
            left, right = normalized.split("_", 1)
            return f"{left}.{right}"
        return normalized

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
            return self._plan_hybrid(summary, metadata, heuristic_plan, architecture_profile)

        return self._plan_ai_first(summary, metadata, heuristic_plan, architecture_profile)

    def _plan_hybrid(
        self,
        summary: str,
        metadata: List[Dict[str, Any]],
        heuristic_plan: Dict[str, Any],
        architecture_profile: str,
    ) -> Dict[str, Any]:
        heur_modules = len(heuristic_plan.get("modules", []))
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": "You improve module plans conservatively and return precise JSON only."},
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
                    - Prefer useful package-style module names like runtime.bot_core, data.storage, features.moderation, features.analytics, features.visuals, features.battle, features.economy, features.progression, commands.general, commands.aura, shared.helpers.
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
        architecture_profile: str,
    ) -> Dict[str, Any]:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": "You produce precise JSON and thoughtful module plans."},
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
                          "name": "package.style_name_or_short_snake_case_name",
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
                    - runtime.bot_core / runtime.events / config.settings / data.storage
                    - commands.general / commands.admin / commands.economy / commands.analytics
                    - features.moderation / features.visuals / features.cards / features.battles / features.tournaments / features.shop / features.leaderboards
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

                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)

                parsed_plan = json.loads(content)
                parsed_plan, sanitation_warnings = self._sanitize_ai_plan(parsed_plan, metadata)
                if sanitation_warnings and self.verbose:
                    for warning in sanitation_warnings:
                        typer.echo(f"AI plan warning: {warning}", err=True)

                parsed_plan = self._complete_missing_segments(parsed_plan, metadata)

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
        if not isinstance(plan, dict):
            return plan
        modules = plan.get("modules", [])
        if not isinstance(modules, list):
            return plan

        valid_segment_ids = {entry["segment_id"] for entry in metadata}
        assigned = {
            seg_id
            for module in modules
            if isinstance(module, dict)
            for seg_id in module.get("segment_ids", [])
            if isinstance(seg_id, str)
        }
        missing = sorted(list(valid_segment_ids - assigned))
        if not missing:
            return plan

        preferred = {"utilities", "utilities_module", "misc", "misc_module"}
        target_idx: Optional[int] = None
        for index, module in enumerate(modules):
            if not isinstance(module, dict):
                continue
            name = str(module.get("name", "")).strip().lower()
            if name in preferred:
                target_idx = index
                break

        filled_module_name = ""
        if target_idx is None:
            existing = {str(module.get("name", "")).strip() for module in modules if isinstance(module, dict)}
            base = "utilities_module"
            name = base
            number = 1
            while name in existing or not name:
                name = f"{base}_{number}"
                number += 1
            filled_module_name = name
            modules.append(
                {
                    "name": name,
                    "description": f"Auto-filled module for {len(missing)} segments omitted by the AI plan.",
                    "segment_ids": missing,
                }
            )
        else:
            module = dict(modules[target_idx])
            filled_module_name = str(module.get("name", "utilities_module"))
            segs = list(module.get("segment_ids", []))
            seg_set = {seg for seg in segs if isinstance(seg, str)}
            for seg_id in missing:
                if seg_id not in seg_set:
                    segs.append(seg_id)
            module["segment_ids"] = segs
            if not module.get("description"):
                module["description"] = "Contains miscellaneous segments (auto-filled)."
            modules[target_idx] = module

        plan["modules"] = modules
        typer.echo(
            f"AI plan omitted {len(missing)} segment(s); assigned them to module {filled_module_name!r}.",
            err=True,
        )
        return plan

    @staticmethod
    def _validate_ai_plan(plan: Dict[str, Any], metadata: List[Dict[str, Any]]) -> Tuple[bool, str]:
        if not isinstance(plan, dict):
            return False, "Plan is not a dictionary"
        if "modules" not in plan or not isinstance(plan.get("modules"), list):
            return False, "Missing or invalid 'modules' field"
        if len(plan["modules"]) == 0:
            return False, "Plan contains no modules"

        valid_segment_ids = {entry["segment_id"] for entry in metadata}
        all_assigned_segments = set()
        seen_names = set()

        for index, module in enumerate(plan["modules"]):
            if not isinstance(module, dict):
                return False, f"Module {index} is not a dict"
            if "name" not in module or not isinstance(module["name"], str) or not module["name"]:
                return False, f"Module {index} has missing or invalid name"
            if "segment_ids" not in module or not isinstance(module["segment_ids"], list):
                return False, f"Module {index} has missing or invalid segment_ids"
            if len(module["segment_ids"]) == 0:
                return False, f"Module {module['name']} is empty (no segments assigned)"
            if module["name"] in seen_names:
                return False, f"Duplicate module name: {module['name']}"
            seen_names.add(module["name"])

            for seg_id in module["segment_ids"]:
                if seg_id not in valid_segment_ids:
                    return False, f"Module {index} references invalid segment: {seg_id}"
                if seg_id in all_assigned_segments:
                    return False, f"Segment {seg_id} assigned to multiple modules (DUPLICATE!)"
                all_assigned_segments.add(seg_id)

        if all_assigned_segments != valid_segment_ids:
            missing = valid_segment_ids - all_assigned_segments
            return False, f"Segments not assigned to any module: {missing}"

        module_deps = LLMPlanner._build_module_dependencies(plan, metadata)
        cycles = LLMPlanner._detect_cycles(module_deps)
        if cycles:
            cycle_info = ", ".join([" -> ".join(cycle) for cycle in cycles])
            return False, f"Circular dependencies detected: {cycle_info}"

        return True, ""

    @staticmethod
    def _sanitize_ai_plan(plan: Dict[str, Any], metadata: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[str]]:
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
                warnings.append(
                    f"Dropped module {module.get('name', '<unknown>')} because it contained no valid segments after sanitization."
                )

        sanitized_plan = {**plan, "modules": sanitized_modules}
        return sanitized_plan, warnings

    @staticmethod
    def _build_module_dependencies(plan: Dict[str, Any], metadata: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
        segment_to_module: Dict[str, str] = {}
        for module in plan.get("modules", []):
            for seg_id in module.get("segment_ids", []):
                segment_to_module[seg_id] = module["name"]

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
                if isinstance(seg_id, str):
                    parts = seg_id.split(":")
                    if len(parts) >= 2:
                        name_to_modules.setdefault(parts[1], set()).add(module_name)

        segment_deps: Dict[str, List[str]] = {}
        for entry in metadata:
            segment_deps[entry["segment_id"]] = entry.get("dependencies", [])

        module_deps: Dict[str, Set[str]] = {module["name"]: set() for module in plan.get("modules", [])}
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
        modules = [module for module in plan.get("modules", []) if isinstance(module, dict)]
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
                    "description": " / ".join(dict.fromkeys(descriptions)) or f"Merged cyclic modules: {', '.join(cycle)}",
                    "segment_ids": merged_segment_ids,
                }
            )

        final_modules = [module for module in modules if str(module.get("name")) not in merged_names]
        final_modules.extend(merged_modules)

        merged_plan = dict(plan)
        merged_plan["modules"] = final_modules
        merged_notes = str(plan.get("notes", "")).strip()
        merged_names_flat = ", ".join(sorted(name for group in cyclic_groups for name in group))
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
        modules = [module for module in plan.get("modules", []) if isinstance(module, dict)]
        total_segments = len(metadata)
        module_count = len(modules)
        if total_segments <= 0:
            return True, ""

        segment_counts = [len([seg for seg in module.get("segment_ids", []) if isinstance(seg, str)]) for module in modules]
        largest_module = max(segment_counts, default=0)
        largest_ratio = largest_module / total_segments

        if total_segments >= 80:
            minimum_useful_modules = min(max(4, requested_max_modules // 2), requested_max_modules)
            if module_count < minimum_useful_modules:
                return False, (
                    f"only {module_count} modules remained for {total_segments} segments "
                    f"(expected at least {minimum_useful_modules})"
                )

            if largest_ratio > 0.55:
                return False, (
                    f"largest module contains {largest_module}/{total_segments} segments "
                    f"({largest_ratio:.0%}), which is too monolithic"
                )

        return True, ""

    @staticmethod
    def _detect_cycles(graph: Dict[str, Set[str]]) -> List[List[str]]:
        cycles: List[List[str]] = []
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        path: List[str] = []

        def dfs(node: str) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_stack:
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
        if not metadata:
            return {"modules": [], "notes": "No segments to process."}

        semantic_keywords = [keyword.strip().lower() for keyword in (semantic_keywords or []) if keyword.strip()]
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

        segments = []
        for entry in metadata:
            segments.append(
                {
                    "id": entry["segment_id"],
                    "kind": entry["kind"],
                    "name": entry["name"],
                    "dependencies": set(entry.get("dependencies", [])),
                    "line": int(entry["lines"].split("-")[0]) if "lines" in entry else 0,
                    "tokens": set(re.findall(r"[a-zA-Z0-9]+", entry["name"].lower())),
                }
            )

        for seg in segments:
            for keyword in semantic_keywords:
                if keyword and keyword in seg["name"].lower():
                    seg["tokens"].add(keyword)

        affinity_graph: Dict[str, Dict[str, int]] = {}
        for segment in segments:
            affinity_graph[segment["id"]] = {}
            for other in segments:
                if other["id"] == segment["id"]:
                    continue

                score = 0
                if other["name"] in segment["dependencies"]:
                    score += 4

                if segment["kind"] == "class" and other["kind"] in {"function", "async_function"}:
                    if segment["name"] in other["dependencies"]:
                        score += 3
                    elif other["name"].startswith(segment["name"].lower() + "_") or other["name"].startswith(segment["name"] + "."):
                        score += 2

                if semantic_grouping:
                    shared = segment["tokens"].intersection(other["tokens"])
                    if shared:
                        score += min(2, len(shared))
                    seg_bucket = LLMPlanner._semantic_bucket(segment["name"])
                    other_bucket = LLMPlanner._semantic_bucket(other["name"])
                    if seg_bucket == other_bucket and seg_bucket != "other":
                        score += 1

                if abs(segment["line"] - other["line"]) <= 40:
                    score += 1

                if score > 0:
                    affinity_graph[segment["id"]][other["id"]] = score

        visited: Set[str] = set()
        groups: List[List[str]] = []
        sorted_segments = sorted(
            segments,
            key=lambda segment: (len(segment["dependencies"]), segment["kind"] == "class", -segment["line"]),
            reverse=True,
        )

        target_modules = min(max_modules, max(1, len(segments) // min_segments_per_module))

        for segment in sorted_segments:
            if segment["id"] in visited:
                continue

            group: Set[str] = set()
            stack = [segment["id"]]

            while stack:
                current_id = stack.pop()
                if current_id in visited:
                    continue

                visited.add(current_id)
                group.add(current_id)

                neighbors = affinity_graph.get(current_id, {})
                ranked_neighbors = sorted(neighbors.items(), key=lambda item: item[1], reverse=True)
                for neighbor_id, weight in ranked_neighbors:
                    if neighbor_id in visited:
                        continue
                    if len(group) >= min_segments_per_module and weight <= 1:
                        continue
                    stack.append(neighbor_id)

            if group:
                groups.append(sorted(list(group)))

        if len(groups) <= 1:
            groups = LLMPlanner._simple_grouping(segments, semantic_keywords)

        groups = LLMPlanner._normalize_groups(
            groups=groups,
            segments=segments,
            target_modules=target_modules,
            min_segments_per_module=min_segments_per_module,
            semantic_keywords=semantic_keywords,
        )

        modules = []
        used_names: Set[str] = set()
        for group_ids in groups:
            group_segments = [segment for segment in segments if segment["id"] in group_ids]
            if not group_segments:
                continue

            primary_segment = max(
                group_segments,
                key=lambda segment: (segment["kind"] == "class", len(segment["dependencies"]), segment["name"]),
            )
            base_name = primary_segment["name"].split("_")[0] or primary_segment["kind"]
            module_name = f"{base_name}_module"

            counter = 1
            original_name = module_name
            while module_name in used_names:
                module_name = f"{original_name}_{counter}"
                counter += 1
            used_names.add(module_name)

            kinds = [segment["kind"] for segment in group_segments]
            kind_counts = {kind: kinds.count(kind) for kind in set(kinds)}
            description_parts = []
            for kind, count in kind_counts.items():
                description_parts.append(f"{count} {kind}{'s' if count > 1 else ''}")
            description = f"Contains {', '.join(description_parts)}"

            modules.append(
                {
                    "name": module_name,
                    "description": description,
                    "segment_ids": group_ids,
                }
            )

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
        semantic_keywords = [keyword.strip().lower() for keyword in (semantic_keywords or []) if keyword.strip()]
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
                    "name": LLMPlanner._package_style_module_name(base_name),
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
                return "models"
            if any(token in lowered for token in ["analyzer", "collector", "parser", "resolver", "scope"]):
                return "analysis"
            if any(token in lowered for token in ["planner", "client"]):
                return "planning"
            if any(token in lowered for token in ["writer", "builder", "exporter", "renderer"]):
                return "writing"
            if any(token in lowered for token in ["validator", "checker", "verifier"]):
                return "writing"
        if any(token in text for token in ["@app.command", "typer.typer", "__main__", "version(", "init_config(", "modularize("]):
            return "cli"
        if any(token in lowered for token in ["analyze", "parse", "collect", "resolve", "dependency"]):
            return "analysis"
        if any(token in lowered for token in ["plan", "group", "sanitize", "cycle"]):
            return "planning"
        if any(token in lowered for token in ["write", "import", "validate", "sort_modules"]):
            return "writing"
        return "shared"

    @staticmethod
    def _tool_cli_plan(
        metadata: List[Dict[str, Any]],
        max_modules: int,
        min_segments_per_module: int,
        semantic_keywords: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        preferred_order = [
            "models",
            "analysis",
            "planning",
            "writing",
            "cli",
            "shared",
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
                _, ids = normalized.pop()
                normalized[-1][1].extend(ids)

        modules = [
            {
                "name": LLMPlanner._package_style_module_name(name),
                "description": f"Tool/CLI architecture module for {name.replace('_', ' ')}.",
                "segment_ids": ids,
            }
            for name, ids in normalized
        ]
        return {
            "modules": modules,
            "notes": "Generated via tool/CLI heuristic planning with package-oriented layered modules.",
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
                _, ids = normalized.pop()
                normalized[-1][1].extend(ids)

        modules = [
            {
                "name": LLMPlanner._package_style_module_name(name),
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
        semantic_keywords = [keyword.strip().lower() for keyword in (semantic_keywords or []) if keyword.strip()]
        functional_groups = {
            "data": [],
            "config": [],
            "main": [],
            "utils": [],
            "domain": [],
            "integration": [],
            "other": [],
        }

        for segment in segments:
            name_lower = segment["name"].lower()
            seg_id = segment["id"]
            if any(keyword in name_lower for keyword in semantic_keywords):
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
        if not groups:
            return groups

        semantic_keywords = [keyword.strip().lower() for keyword in (semantic_keywords or []) if keyword.strip()]
        id_to_segment = {segment["id"]: segment for segment in segments}

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

        def similarity(group_a: List[str], group_b: List[str]) -> int:
            shared = group_signature(group_a).intersection(group_signature(group_b))
            return len(shared)

        normalized = [list(dict.fromkeys(group)) for group in groups]

        changed = True
        while changed:
            changed = False
            tiny_idx = next(
                (
                    index
                    for index, group in enumerate(normalized)
                    if len(group) < min_segments_per_module and len(normalized) > 1
                ),
                None,
            )
            if tiny_idx is None:
                break
            tiny = normalized[tiny_idx]
            best_idx = None
            best_score = -1
            for index, group in enumerate(normalized):
                if index == tiny_idx:
                    continue
                score = similarity(tiny, group)
                if score > best_score:
                    best_score = score
                    best_idx = index
            if best_idx is not None:
                normalized[best_idx] = list(dict.fromkeys(normalized[best_idx] + tiny))
                normalized.pop(tiny_idx)
                changed = True

        while len(normalized) > target_modules and len(normalized) > 1:
            best_pair = None
            best_score = -1
            for left in range(len(normalized)):
                for right in range(left + 1, len(normalized)):
                    score = similarity(normalized[left], normalized[right])
                    if score > best_score:
                        best_score = score
                        best_pair = (left, right)
            if best_pair is None:
                break
            left, right = best_pair
            normalized[left] = list(dict.fromkeys(normalized[left] + normalized[right]))
            normalized.pop(right)

        return normalized



