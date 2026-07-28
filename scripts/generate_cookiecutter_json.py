#!/usr/bin/env python3
"""Generate cookiecutter.json files for each sub-template from a common definition.

This script DRYs the 9 common cookiecutter variables and 9 common __prompts__
that are duplicated across all 6 sub-templates (cuda, cxx, go, py, rs, ts).

Usage:
    python scripts/generate_cookiecutter_json.py

The per-template overrides are defined in the TEMPLATE_CONFIGS dict below.
The script writes cookiecutter.json to each `template/<type>/` directory.
"""

import json
import logging
from pathlib import Path
from typing import Any

# Repository root
ROOT = Path(__file__).resolve().parent.parent

# JSON indentation width for dict values (like __prompts__)
JSON_INDENT = 4

# Common variables shared by ALL sub-templates
COMMON_VARS = {
    "project_name": "My Awesome Project",
    "project_slug": "{{ cookiecutter.project_name | slugify }}",
    "package_name": "{{ cookiecutter.project_slug | replace('-', '_') }}",
    "project_desc": "A nice project",
    "project_version": "0.0.1",
    "email": "email@example.com",
    "full_name": "Firstname Lastname",
    "github_username": "your-org-or-username",
    # Computed/derived variable
    "__gh_slug": "{{ cookiecutter.github_username }}/{{ cookiecutter.project_slug }}",
}

# Common __prompts__ shared by ALL sub-templates
COMMON_PROMPTS = {
    "project_name": "Enter your project name",
    "project_slug": "Generated project identifier (slug)",
    "package_name": "Generated package/module name",
    "project_desc": "Brief project description",
    "project_version": "Initial project version",
    "email": "Author's email address",
    "full_name": "Author's full name",
    "github_username": "GitHub username or organization",
}

# Per-template configurations
# Each entry specifies:
#   - desc: project description (overrides common)
#   - extra_vars: additional cookiecutter variables specific to this template
#   - extra_prompts: additional __prompts__ specific to this template
#   - copy_without_render: list of paths to exclude from rendering (optional)
TEMPLATE_CONFIGS = {
    "cuda": {
        "desc": "A nice cuda project",
        "extra_vars": {
            "cxx_build_tool": ["cmake", "xmake"],
            "cuda_standard_version": "17",
            "cxx_standard_version": "20",
        },
    },
    "cxx": {
        "desc": "A nice cxx project",
        "extra_vars": {
            "cxx_build_tool": ["cmake", "xmake"],
            "cxx_standard_version": "20",
        },
        "copy_without_render": ["third_party"],
    },
    "go": {
        "desc": "A nice go project",
        "extra_vars": {
            "go_version": "1.22",
        },
    },
    "py": {
        "desc": "A nice python project",
        "extra_vars": {
            "python_version": "3.13",
        },
    },
    "rs": {
        "desc": "A nice rust project",
        "extra_vars": {
            "crate_type": ["bin", "lib"],
        },
    },
    "ts": {
        "desc": "A nice typescript project",
        "extra_vars": {
            "node_version": "20",
        },
    },
}


def build_cookiecutter_json(template_name: str) -> dict[str, Any]:
    """Build the cookiecutter.json dict for a given template.

    Key ordering matches the original cookiecutter.json convention:
    1. Common variables (except __gh_slug)
    2. Template-specific variables
    3. __gh_slug (computed)
    4. __prompts__
    5. _copy_without_render (if applicable)
    """
    config = TEMPLATE_CONFIGS[template_name]

    # Separate __gh_slug (computed) from other common vars for ordering
    common_public = {k: v for k, v in COMMON_VARS.items() if k != "__gh_slug"}

    # 1. Common variables (except __gh_slug)
    result = dict(common_public)
    # Override the project_desc for this template
    result["project_desc"] = config["desc"]
    # 2. Template-specific variables
    result.update(config["extra_vars"])
    # 3. __gh_slug (computed, placed after template-specific vars)
    result["__gh_slug"] = COMMON_VARS["__gh_slug"]
    # 4. __prompts__: common prompts + any template-specific prompts
    result["__prompts__"] = dict(COMMON_PROMPTS)
    # 5. _copy_without_render (if applicable)
    if "copy_without_render" in config:
        result["_copy_without_render"] = config["copy_without_render"]

    return result


def format_cookiecutter_json(data: dict[str, Any], template_name: str) -> str:
    """Format cookiecutter.json with blank lines between variable groups.

    Groups:
      1. Common variables (project_name through github_username)
      2. Template-specific variables
      3. __gh_slug (computed)
      4. __prompts__
      5. _copy_without_render (if applicable)
    """
    lines = ["{"]

    config = TEMPLATE_CONFIGS[template_name]
    extra_keys = list(config.get("extra_vars", {}).keys())

    # Build key ordering: common, extra, __gh_slug, __prompts__, _copy_without_render
    all_ordered_keys = []
    # Common vars (excluding __gh_slug, which goes later)
    for k in COMMON_VARS:
        if k == "__gh_slug":
            continue
        if k == "project_desc":
            all_ordered_keys.append(k)  # value overridden below
        else:
            all_ordered_keys.append(k)
    # Extra vars
    all_ordered_keys.extend(extra_keys)
    # __gh_slug
    all_ordered_keys.append("__gh_slug")
    # __prompts__
    all_ordered_keys.append("__prompts__")
    # _copy_without_render
    if "copy_without_render" in config:
        all_ordered_keys.append("_copy_without_render")

    # Insert blank lines between groups
    group_boundaries = {
        extra_keys[0] if extra_keys else "__gh_slug",  # before extra vars or __gh_slug
        "__gh_slug",  # before __gh_slug
    }

    for i, key in enumerate(all_ordered_keys):
        if i > 0 and key in group_boundaries:
            lines.append("")
        val = data[key]
        if isinstance(val, dict):
            # Dict values (like __prompts__) get multi-line pretty formatting
            formatted = json.dumps(val, indent=JSON_INDENT, ensure_ascii=False)
            formatted = formatted.replace("\n", "\n    ")
            comma = "" if i == len(all_ordered_keys) - 1 else ","
            lines.append(f'    "{key}": {formatted}{comma}')
        elif isinstance(val, list):
            # List values (like choices) stay on a single line
            comma = "" if i == len(all_ordered_keys) - 1 else ","
            lines.append(f'    "{key}": {json.dumps(val, ensure_ascii=False)}{comma}')
        else:
            comma = "" if i == len(all_ordered_keys) - 1 else ","
            lines.append(f'    "{key}": {json.dumps(val, ensure_ascii=False)}{comma}')

    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Generate cookiecutter.json for all sub-templates.

    Exits with an error if any file cannot be written.
    """
    for template_name in TEMPLATE_CONFIGS:
        data = build_cookiecutter_json(template_name)
        content = format_cookiecutter_json(data, template_name)
        out_path = ROOT / "template" / template_name / "cookiecutter.json"
        try:
            out_path.write_text(content)
        except OSError as exc:
            logging.error("Error writing %s: %s", out_path, exc)
            raise SystemExit(1) from exc
        logging.info("Generated %s", out_path)

    logging.info(
        "Generated cookiecutter.json for %d sub-templates.",
        len(TEMPLATE_CONFIGS),
    )
    logging.info(
        "__gh_slug is computed from github_username and project_slug."
    )
    logging.info(
        "To regenerate, run: python scripts/generate_cookiecutter_json.py"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
