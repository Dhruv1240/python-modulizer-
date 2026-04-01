# Modularizer

A Python tool for splitting large Python files into smaller, cohesive modules.

`modulizer.py` is the core command-line engine.
`modulizer_gui.py` provides a desktop interface.

## Key Features

- AI-assisted module planning using an OpenAI-compatible backend.
- Offline heuristic mode when no API key is available.
- AST-based dependency analysis and smart import handling.
- Cross-module imports for generated modules.
- Validation that generated files can be imported.
- JSON config support and CLI overrides.

## Requirements

- Python 3.8+
- `openai`
- `typer`

## Install

```bash
pip install openai typer
```

## Run the GUI

```bash
python modulizer_gui.py
```

The GUI lets you configure:
- Input file and output directory
- Model, API key, and base URL
- Offline and heuristic fallbacks
- Semantic grouping and keywords
- Retry limits and validation mode

## Run the CLI

### Modularize a file

```bash
python modulizer.py modularize --input-file my_script.py --output-dir modules
```

### Use offline heuristic mode

```bash
python modulizer.py modularize --input-file my_script.py --output-dir modules --offline
```

### Generate a config template

```bash
python modulizer.py init-config
```

### Show version

```bash
python modulizer.py version
```

## Command Options

### `modularize`

Required:
- `--input-file PATH`
- `--output-dir PATH`

Common options:
- `--model TEXT`
- `--api-key TEXT`
- `--openai-base-url TEXT`
- `--temperature FLOAT`
- `--top-p FLOAT`
- `--top-k INT`
- `--frequency-penalty FLOAT`
- `--offline`
- `--heuristic-fallback`
- `--config PATH`
- `--verbose`
- `--strict-validation`

### `init-config`

Create a starter config file:

```bash
python modulizer.py init-config --output-file modulizer_config.json
```

### `version`

```bash
python modulizer.py version
```

## Configuration File

Example config:

```json
{
  "model": "gemini-2.5-flash",
  "api_key": "<YOUR_API_KEY>",
  "openai_base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
  "temperature": 0.9,
  "top_p": 0.3,
  "top_k": 20,
  "frequency_penalty": 0.8,
  "verbose": false,
  "max_modules": 8,
  "min_segments_per_module": 2,
  "semantic_grouping": true,
  "semantic_keywords": [],
  "ai_retries": 5,
  "strict_validation": false,
  "heuristic_fallback": false
}
```

## Examples

### AI-powered modularization

```bash
export OPENAI_API_KEY="your_key_here"
python modulizer.py modularize --input-file app.py --output-dir modules --verbose
```

### Offline-only modularization

```bash
python modulizer.py modularize --input-file app.py --output-dir modules --offline
```

### Fallback when AI fails

```bash
python modulizer.py modularize --input-file app.py --output-dir modules --heuristic-fallback
```

## Output

The tool writes:

- generated module files
- `__init__.py`
- `module_plan.json`

`module_plan.json` records the generated plan and segment mappings.

## How It Works

1. Parse the source file with the Python AST.
2. Identify top-level blocks, classes, functions, and imports.
3. Build a module plan using AI or heuristic fallback.
4. Write each module with only the imports it requires.
5. Validate generated modules by importing them.

## Troubleshooting

- Use `--offline` when no API key is available.
- Use `--heuristic-fallback` to fall back to heuristic planning after AI errors.
- If validation fails, inspect `module_plan.json` and generated imports.
