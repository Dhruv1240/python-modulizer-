from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import typer
from typer.models import OptionInfo

from .analysis import SourceAnalyzer
from .planning import LLMPlanner
from .writing import ModuleWriter

app = typer.Typer(
    help="Split a large Python file into AI-planned modules.",
    add_completion=False,
)


@app.command()
def version() -> None:
    """Show version information."""
    typer.echo("pyfract 0.1.0")


@app.command()
def init_config(output_file: Path = typer.Option("pyfract_config.json", help="Path to create config file.")) -> None:
    """Generate a sample configuration file."""
    sample_config = {
        "model": "",
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
    except IOError as exc:
        typer.echo(f"Error: Failed to write config file: {exc}", err=True)
        raise typer.Exit(1)


@app.command()
def modularize(
    input_file: Path = typer.Option(..., exists=True, readable=True, help="Python file to split."),
    output_dir: Path = typer.Option(..., help="Directory to write modules."),
    model: Optional[str] = typer.Option(
        None,
        help=(
            "Chat model name for the configured LLM API. Supports any provider-specific "
            "model id. If omitted, MODULIZER_MODEL / OPENAI_MODEL / OPENROUTER_MODEL / "
            "LLM_MODEL is used."
        ),
    ),
    api_key: Optional[str] = typer.Option(None, help="API key (overrides OPENAI_API_KEY)."),
    offline: bool = typer.Option(False, help="Skip AI entirely; use heuristic plan only."),
    openai_base_url: Optional[str] = typer.Option(
        None,
        help=f"API base URL (overrides OPENAI_BASE_URL; default {LLMPlanner.DEFAULT_BASE_URL!r}).",
    ),
    planning_mode: Optional[str] = typer.Option(None, help="Planning mode: safe, hybrid, or ai_first."),
    temperature: Optional[float] = typer.Option(None, min=0.0, max=2.0, help="Sampling temperature (default 0.9)."),
    top_p: Optional[float] = typer.Option(None, min=0.0, max=1.0, help="Nucleus sampling top_p (default 0.3)."),
    top_k: Optional[int] = typer.Option(None, min=1, help="Top-k sampling (default 20)."),
    frequency_penalty: Optional[float] = typer.Option(None, min=-2.0, max=2.0, help="Frequency penalty (default 0.8)."),
    config: Optional[Path] = typer.Option(None, help="JSON configuration file to load default settings from."),
    max_modules: Optional[int] = typer.Option(None, min=1, max=64, help="Target maximum number of output modules."),
    min_segments_per_module: Optional[int] = typer.Option(None, min=1, max=20, help="Minimum segments per module target."),
    semantic_grouping: Optional[bool] = typer.Option(None, "--semantic-grouping/--no-semantic-grouping", help="Use semantic naming signals to keep related symbols together."),
    semantic_keywords: Optional[str] = typer.Option(None, help="Comma-separated domain keywords to bias semantic grouping."),
    strict_validation: Optional[bool] = typer.Option(None, "--strict-validation/--no-strict-validation", help="Fail the run if validation reports issues."),
    ai_retries: Optional[int] = typer.Option(None, min=1, max=15, help="Maximum AI planning retries (default 5)."),
    heuristic_fallback: Optional[bool] = typer.Option(None, "--heuristic-fallback/--no-heuristic-fallback", help="If AI fails after retries, fall back to heuristic plan."),
    verbose: bool = typer.Option(False, help="Enable verbose output."),
) -> None:
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

    config_data = {}
    if config and config.exists():
        try:
            config_data = json.loads(config.read_text())
        except (json.JSONDecodeError, IOError) as exc:
            typer.echo(f"Warning: Failed to load config file {config}: {exc}", err=True)

    model = model or config_data.get("model") or LLMPlanner.resolve_model()
    api_key = api_key or config_data.get("api_key")
    openai_base_url = openai_base_url or config_data.get("openai_base_url")
    planning_mode = str(planning_mode or config_data.get("planning_mode") or "safe").strip().lower()
    temperature = temperature if temperature is not None else float(config_data.get("temperature", 0.9))
    top_p = top_p if top_p is not None else float(config_data.get("top_p", 0.3))
    top_k = top_k if top_k is not None else int(config_data.get("top_k", 20))
    frequency_penalty = frequency_penalty if frequency_penalty is not None else float(config_data.get("frequency_penalty", 0.8))
    verbose = verbose or config_data.get("verbose", False)
    max_modules = max_modules or config_data.get("max_modules", 8)
    min_segments_per_module = min_segments_per_module or config_data.get("min_segments_per_module", 2)
    semantic_grouping = semantic_grouping if semantic_grouping is not None else bool(config_data.get("semantic_grouping", True))
    strict_validation = strict_validation if strict_validation is not None else bool(config_data.get("strict_validation", False))
    ai_retries = ai_retries or config_data.get("ai_retries", 5)
    heuristic_fallback = heuristic_fallback if heuristic_fallback is not None else bool(config_data.get("heuristic_fallback", True))
    if planning_mode in {"ai-first", "ai first"}:
        planning_mode = "ai_first"
    if planning_mode not in LLMPlanner.PLANNING_MODES:
        typer.echo(f"Warning: Unknown planning mode {planning_mode!r}; using 'safe' instead.", err=True)
        planning_mode = "safe"
    if planning_mode == "safe":
        offline = True
        heuristic_fallback = True
    elif planning_mode == "hybrid":
        heuristic_fallback = True

    raw_keywords = semantic_keywords if semantic_keywords is not None else config_data.get("semantic_keywords", [])
    if isinstance(raw_keywords, str):
        semantic_keywords_list = [keyword.strip().lower() for keyword in raw_keywords.split(",") if keyword.strip()]
    elif isinstance(raw_keywords, list):
        semantic_keywords_list = [str(keyword).strip().lower() for keyword in raw_keywords if str(keyword).strip()]
    else:
        semantic_keywords_list = []

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
    except SyntaxError as exc:
        typer.echo(f"Error: Invalid Python syntax in {input_file}: {exc}", err=True)
        raise typer.Exit(1)
    except Exception as exc:
        typer.echo(f"Error: Failed to analyze {input_file}: {exc}", err=True)
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
    except Exception as exc:
        typer.echo(f"Error: Failed to generate plan: {exc}", err=True)
        typer.echo("Use --heuristic-fallback to allow heuristic planning after AI failure, or --offline for heuristic-only.", err=True)
        raise typer.Exit(1)

    if verbose:
        typer.echo(f"Writing {len(plan.get('modules', []))} modules to {output_dir}...", err=True)
    writer = ModuleWriter()
    try:
        source_code = input_file.read_text(encoding="utf-8")
        manifest = writer.write(plan, segments, output_dir, input_file.name, source_code, strict_validation=strict_validation)
        typer.echo(f"Modules written. Manifest: {manifest}")
    except Exception as exc:
        typer.echo(f"Error: Failed to write modules: {exc}", err=True)
        raise typer.Exit(1)



