# Pyfract

Turn one large Python file into a cleaner Python package with either a GUI or a CLI.

Pyfract is a better, more practical version of the tool for people who want to break up huge scripts without doing the whole split by hand. It analyzes a single-file project, groups code into modules, writes the package, and validates the result so you can catch obvious issues early.

## Why Pyfract

- Built for large Python files that became hard to manage
- Supports both GUI and CLI workflows
- Gives you a fast first-pass modularization instead of a full manual rewrite
- Tries to reduce broken outputs with validation after generation
- Made to be beginner-friendly, especially through the GUI flow

## Beginner-Friendly Design

One of the main goals of this tool is to make modularization easier for beginners.

- The GUI helps users run the tool without needing to remember long commands
- The planning modes are simple to understand: `safe`, `hybrid`, and `ai_first`
- The generated `module_plan.json` makes it easier to inspect what the tool decided
- The safest default path is clear: start with `safe`, review the result, then improve manually if needed

If you are new to code organization, refactoring, or package structure, Pyfract is meant to give you a much easier starting point.

## Highlights

- Desktop GUI for normal users: [`modulizer_gui.py`](c:\modulizer\modulizer_gui.py)
- CLI for scripting and repeatable runs: [`modulizer.py`](c:\modulizer\modulizer.py)
- Three planning modes: `safe`, `hybrid`, `ai_first`
- Feature-based grouping for large multi-purpose files
- Validation after generation
- `module_plan.json` output for reviewing the final grouping

## Best Use Case

Pyfract works best when you have:

- a very large `.py` file
- many top-level functions, classes, and constants
- a script, bot, or utility that grew into a monolith
- a project where you want a strong starting split before manual cleanup

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
- Best option if you want the fewest bugs and the most stable output

### `hybrid`

Heuristics first, AI refinement second.

- Starts with the safer heuristic split
- Lets AI improve naming and grouping when possible
- Falls back more safely than pure AI-first planning

### `ai_first`

AI planning before anything else.

- Can sometimes produce a cleaner architecture
- More experimental and less predictable
- Best used when you are okay with rough edges

## Quick Start

### GUI

```powershell
python modulizer_gui.py
```

Recommended settings for large files:

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
- generated subpackages when the planner chooses a nested structure
- `__init__.py`
- `module_plan.json`

Example output:

```text
modules/
  __init__.py
  commands/
    __init__.py
    general.py
    aura.py
  data/
    __init__.py
    storage.py
  features/
    __init__.py
    analytics.py
    battle.py
    economy.py
    moderation.py
    progression.py
    visuals.py
  runtime/
    __init__.py
    bot_core.py
  shared/
    __init__.py
    helpers.py
  module_plan.json
```

## How It Works

1. Parse the source file into top-level segments.
2. Analyze names, symbols, and dependencies.
3. Build a module plan.
4. Write the generated modules.
5. Validate the generated package.

## Validation

Generated modules are validated after writing.

Current validation includes:

- syntax parsing
- bytecode compilation
- package-relative import target checks
- optional runtime import validation when enabled externally

This catches many structural mistakes, but it does not guarantee perfect runtime behavior. You should still test the output in your real project.

## Bugs And Limitations

This tool is better than doing a full manual split from scratch, but it is not bug-free.

- Files with heavy cross-dependencies can still be hard to split correctly
- Some generated packages may need manual cleanup
- Validation is better at catching structure problems than behavior problems
- AI-based planning can still make weak or awkward grouping decisions
- `ai_first` is the most experimental mode and may produce the most unstable results
- Large or unusual projects may still need follow-up edits after generation

So yes, the tool helps a lot, but there can still be bugs, edge cases, and imperfect module boundaries. The safest approach is to treat the output as a strong first draft instead of a perfect final architecture.

## Reliability Notes

If your goal is to reduce bugs as much as possible:

1. Start with `safe`
2. Try `hybrid` if you want some AI help
3. Use `ai_first` only for experiments

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

The GUI is designed to be more friendly for normal users and beginners. It includes:

- input and output folder pickers
- a planning mode selector
- generated modules location display
- buttons to open or copy the output path
- command preview for the equivalent CLI command

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

## Final Note

Pyfract is meant to be a better and more beginner-friendly version of this kind of tool. It helps automate one of the most annoying parts of refactoring large Python files, while still being honest about the fact that bugs, edge cases, and manual cleanup can still happen.
