# Modulizer

Split a large Python file into a structured Python package with a GUI or CLI.

Modulizer is built for big single-file scripts and bots that have become hard to navigate, maintain, or refactor manually. It analyzes the file, plans module boundaries, writes the generated package, and validates the output so you can catch bad splits early.

## Highlights

- Desktop GUI for normal users: [`modulizer_gui.py`](c:\modulizer\modulizer_gui.py)
- CLI for scripting and repeatable runs: [`modulizer.py`](c:\modulizer\modulizer.py)
- Three planning modes: `safe`, `hybrid`, `ai_first`
- Feature-oriented splitting for large multi-purpose files
- Output validation after generation
- `module_plan.json` manifest for inspecting the final grouping

## Best Use Case

Modulizer is most useful when you have:

- a very large `.py` file
- lots of top-level functions, classes, and constants
- a bot, utility app, or script that grew into a monolith
- a project where you want a strong first-pass split before manual cleanup

## Planning Modes

| Mode | Best for | Reliability | AI required |
| --- | --- | --- | --- |
| `safe` | everyday use | highest | no |
| `hybrid` | cautious AI assistance | high | optional |
| `ai_first` | experimentation | lowest | yes |

### `safe`

Heuristic feature-based planning only.

- Most reliable mode
- No AI dependency
- Best choice when you care more about fewer broken outputs than fancy architecture

### `hybrid`

Heuristics first, AI refinement second.

- Starts with the safer heuristic split
- Lets AI improve grouping and naming only when possible
- Falls back safely if AI is unavailable or weak

### `ai_first`

AI planning before anything else.

- Can sometimes produce a nicer architecture
- Less predictable than `safe` or `hybrid`
- Best treated as an experimental mode

## Quick Start

### GUI

Run:

```powershell
python modulizer_gui.py
```

Recommended GUI settings for large files:

- Planning mode: `safe`
- Max modules: `12` to `16`
- Min segments per module: `3`

### CLI

Safe mode:

```powershell
python modulizer.py modularize --input-file bot.py --output-dir modules --planning-mode safe
```

Hybrid mode:

```powershell
python modulizer.py modularize --input-file bot.py --output-dir modules --planning-mode hybrid
```

AI-first mode:

```powershell
python modulizer.py modularize --input-file bot.py --output-dir modules --planning-mode ai_first
```

## What Gets Generated

The output folder becomes a Python package containing:

- generated module files
- `__init__.py`
- `module_plan.json`

Example output for a large Discord bot:

```text
modules/
  __init__.py
  analytics.py
  aura_commands.py
  battle.py
  bot_core.py
  commands_general.py
  data_storage.py
  economy.py
  moderation.py
  progression.py
  shared.py
  visuals.py
  module_plan.json
```

## How It Works

1. Parse the source file into top-level segments.
2. Analyze names, symbols, and dependencies.
3. Build a module plan.
4. Write the generated modules.
5. Validate the output package.

## Validation

Generated modules are validated after writing.

Current validation includes:

- syntax parsing
- bytecode compilation
- package-relative import target checks
- optional runtime import validation when enabled externally

This catches many structural mistakes, but it does not guarantee full behavioral correctness. For important projects, treat the generated output as a strong first draft and still run your real tests or a runtime smoke test.

## Reliability Notes

If your goal is “as few broken generated modules as possible”:

1. use `safe` first
2. try `hybrid` if you want AI help
3. use `ai_first` only when you are willing to trade reliability for experimentation

In general:

- `safe` is the most reliable
- `hybrid` is the best balance
- `ai_first` is the least predictable

## Useful CLI Options

### Core

- `--planning-mode safe|hybrid|ai_first`
- `--input-file`
- `--output-dir`
- `--max-modules`
- `--min-segments-per-module`
- `--strict-validation / --no-strict-validation`
- `--verbose`

### Grouping

- `--semantic-grouping / --no-semantic-grouping`
- `--semantic-keywords`

### AI

- `--model`
- `--api-key`
- `--openai-base-url`
- `--temperature`
- `--top-p`
- `--top-k`
- `--frequency-penalty`
- `--ai-retries`
- `--heuristic-fallback / --no-heuristic-fallback`

## Config File

Create a sample config:

```powershell
python modulizer.py init-config --output-file modulizer_config.json
```

Run using that config:

```powershell
python modulizer.py modularize --input-file bot.py --output-dir modules --config modulizer_config.json
```

## GUI Notes

The GUI is designed to be friendlier for normal users. It includes:

- input and output folder pickers
- a planning mode selector
- generated modules location display
- buttons to open or copy the output path
- command preview for equivalent CLI usage

## Screenshots

You can add screenshots here later, for example:

- main GUI window
- output folder after generation
- example `module_plan.json`

```markdown
![Main GUI](docs/gui-main.png)
![Generated Modules](docs/generated-modules.png)
```

## Project Files

- [`modulizer.py`](c:\modulizer\modulizer.py): CLI and planning engine
- [`modulizer_gui.py`](c:\modulizer\modulizer_gui.py): desktop GUI
- [`build_exe.ps1`](c:\modulizer\build_exe.ps1): packaging helper

## Recommendations

- Start with `safe`
- Increase `max_modules` for very large files
- Keep `min_segments_per_module` small enough to allow useful splitting
- Review `module_plan.json` after each run
- Run your own tests after generation if correctness matters

## Limitations

- Files with many global cross-dependencies are hard to split perfectly
- Some projects need manual cleanup after generation
- Validation catches structure issues better than runtime behavior issues
- AI planning can still produce weak plans, which is why `safe` and `hybrid` exist
