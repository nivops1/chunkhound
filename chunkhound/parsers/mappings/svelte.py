"""Svelte language mapping for unified parser architecture.

This module provides Svelte component-specific parsing that handles the multi-section
structure of Svelte Single File Components (.svelte files).

## Approach
- Extract <script>, markup, <style> sections via regex
- Parse script content with TypeScript parser (inherited)
- Create text chunks for markup and style sections
- Add Svelte-specific metadata for reactive declarations and stores

## Supported Features
- <script lang="ts"> parsing
- Reactive declarations ($: syntax)
- Store usage detection ($storeName pattern)
- Regular <script> support
- Markup as searchable text block
- Style as optional text block

## Limitations (Phase 1)
- Markup directives not fully parsed (no #if/#each structure)
- No cross-section reference tracking
- No component usage graph
- Basic section extraction (regex-based)
"""

import re
from typing import TYPE_CHECKING, Any

from chunkhound.core.types.common import Language
from chunkhound.parsers.mappings.typescript import TypeScriptMapping

if TYPE_CHECKING:
    from tree_sitter import Node as TSNode

try:
    from tree_sitter import Node as TSNode

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    TSNode = Any  # type: ignore


class SvelteMapping(TypeScriptMapping):
    """Svelte component language mapping extending TypeScript mapping.

    Handles Svelte Single File Component structure with multiple sections.
    Script sections are parsed as TypeScript, markup and style as text.
    """

    def __init__(self) -> None:
        """Initialize Svelte mapping (delegates to TypeScript for script parsing)."""
        super().__init__()
        self.language = Language.SVELTE  # Override to SVELTE

    # Section extraction patterns
    SCRIPT_PATTERN = re.compile(
        r"<script\s*([^>]*)>(.*?)</script>", re.DOTALL | re.IGNORECASE
    )

    STYLE_PATTERN = re.compile(
        r"<style\s*([^>]*)>(.*?)</style>", re.DOTALL | re.IGNORECASE
    )

    # Svelte-specific patterns
    REACTIVE_DECLARATION_PATTERN = re.compile(r"\$:\s*")
    STORE_USAGE_PATTERN = re.compile(r"\$\w+")

    def extract_sections(self, content: str) -> dict[str, list[tuple[str, str, int]]]:
        """Extract script, markup, and style sections from Svelte component.

        Args:
            content: Full Svelte component content

        Returns:
            Dictionary with 'script', 'markup', 'style' keys, each containing
            list of (attributes, section_content, start_line) tuples
        """
        sections: dict[str, list[tuple[str, str, int]]] = {
            "script": [],
            "markup": [],
            "style": [],
        }

        # Extract script sections
        for match in self.SCRIPT_PATTERN.finditer(content):
            attrs = match.group(1).strip()
            script_content = match.group(2)
            start_line = content[: match.start()].count("\n") + 1
            sections["script"].append((attrs, script_content, start_line))

        # Extract style sections
        for match in self.STYLE_PATTERN.finditer(content):
            attrs = match.group(1).strip()
            style_content = match.group(2)
            start_line = content[: match.start()].count("\n") + 1
            sections["style"].append((attrs, style_content, start_line))

        # Extract markup (everything outside script and style tags)
        markup_content = content
        # Remove script tags
        markup_content = self.SCRIPT_PATTERN.sub("", markup_content)
        # Remove style tags
        markup_content = self.STYLE_PATTERN.sub("", markup_content)

        # Only add markup if there's meaningful content
        stripped = markup_content.strip()
        if stripped:
            sections["markup"].append(("", stripped, 1))

        return sections

    def get_script_lang(self, attrs: str) -> str:
        """Extract lang attribute from script tag (ts, js).

        Args:
            attrs: Script tag attributes string

        Returns:
            Language identifier (ts, js, etc.)
        """
        lang_match = re.search(r'lang\s*=\s*["\']?(\w+)["\']?', attrs, re.IGNORECASE)
        if lang_match:
            return lang_match.group(1).lower()
        return "js"  # Default to JavaScript

    def is_module_script(self, attrs: str) -> bool:
        """Check if script tag has 'context="module"' attribute.

        Svelte uses context="module" for module-level code that runs once
        when the module first evaluates.

        Args:
            attrs: Script tag attributes string

        Returns:
            True if this is a module script
        """
        return 'context="module"' in attrs or "context='module'" in attrs

    def detect_reactive_declarations(self, script_content: str) -> list[str]:
        """Detect reactive declarations ($: syntax) in script.

        Svelte's reactive declarations automatically re-run when dependencies change.

        Args:
            script_content: Script section content

        Returns:
            List of reactive declaration lines
        """
        reactive_declarations = []
        for line in script_content.split("\n"):
            if self.REACTIVE_DECLARATION_PATTERN.search(line):
                reactive_declarations.append(line.strip())
        return reactive_declarations

    def detect_store_usage(self, content: str) -> list[str]:
        """Detect Svelte store usage ($storeName pattern).

        Svelte auto-subscribes to stores prefixed with $.

        Args:
            content: Script or markup content

        Returns:
            List of unique store identifiers (without $ prefix)
        """
        matches = self.STORE_USAGE_PATTERN.findall(content)
        # Remove $ prefix and filter out reactive declarations ($:)
        stores = [match[1:] for match in matches if match != "$:"]
        return list(set(stores))  # Remove duplicates

    def detect_svelte_directives(self, markup_content: str) -> list[str]:
        """Detect common Svelte directives in markup.

        Args:
            markup_content: Markup section content

        Returns:
            List of directive types found (e.g., ['if', 'each', 'await'])
        """
        directives = []
        directive_patterns = {
            "if": re.compile(r"\{#if\s+"),
            "each": re.compile(r"\{#each\s+"),
            "await": re.compile(r"\{#await\s+"),
            "key": re.compile(r"\{#key\s+"),
        }

        for directive_name, pattern in directive_patterns.items():
            if pattern.search(markup_content):
                directives.append(directive_name)

        return directives
