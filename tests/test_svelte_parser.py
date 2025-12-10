"""Tests for Svelte component parser."""

import pytest
from pathlib import Path

from chunkhound.core.types.common import FileId, Language
from chunkhound.parsers.mappings.svelte import SvelteMapping


class TestSvelteMapping:
    """Test SvelteMapping section extraction and Svelte-specific features."""

    def test_extract_script_section(self):
        """Test extracting script section from Svelte component."""
        content = '''
<script lang="ts">
    let count = 0;
</script>
        '''
        mapping = SvelteMapping()
        sections = mapping.extract_sections(content)

        assert len(sections['script']) == 1
        attrs, script_content, start_line = sections['script'][0]
        assert 'lang="ts"' in attrs
        assert 'let count = 0' in script_content

    def test_extract_multiple_sections(self):
        """Test extracting script, markup, and style sections."""
        content = '''
<script>
    let name = 'world';
</script>

<main>
    <h1>Hello {name}!</h1>
</main>

<style>
    h1 { color: purple; }
</style>
        '''
        mapping = SvelteMapping()
        sections = mapping.extract_sections(content)

        assert len(sections['script']) == 1
        assert len(sections['markup']) == 1
        assert len(sections['style']) == 1
        assert 'let name' in sections['script'][0][1]
        assert 'Hello {name}' in sections['markup'][0][1]
        assert 'color: purple' in sections['style'][0][1]

    def test_detect_reactive_declarations(self):
        """Test detecting Svelte reactive declarations ($: syntax)."""
        script = '''
let count = 0;
$: doubled = count * 2;
$: console.log('count is', count);
$: {
    console.log('complex reactive statement');
}
        '''
        mapping = SvelteMapping()
        reactive_declarations = mapping.detect_reactive_declarations(script)

        assert len(reactive_declarations) >= 2
        assert any('$: doubled' in decl for decl in reactive_declarations)
        assert any('$: console.log' in decl for decl in reactive_declarations)

    def test_detect_store_usage(self):
        """Test detecting Svelte store usage ($storeName pattern)."""
        content = '''
<script>
    import { count } from './stores';
</script>

<main>
    <h1>Count: {$count}</h1>
    <button on:click={() => $count++}>Increment</button>
    <p>Double: {$doubled}</p>
</main>
        '''
        mapping = SvelteMapping()
        stores = mapping.detect_store_usage(content)

        assert 'count' in stores
        assert 'doubled' in stores
        # Should not include $: reactive declarations
        assert ':' not in ''.join(stores)

    def test_detect_svelte_directives(self):
        """Test detecting Svelte control flow directives."""
        markup = '''
<main>
    {#if user.loggedIn}
        <p>Welcome {user.name}!</p>
    {/if}

    {#each items as item}
        <p>{item}</p>
    {/each}

    {#await promise}
        <p>Loading...</p>
    {:then value}
        <p>Result: {value}</p>
    {:catch error}
        <p>Error: {error}</p>
    {/await}

    {#key value}
        <Component />
    {/key}
</main>
        '''
        mapping = SvelteMapping()
        directives = mapping.detect_svelte_directives(markup)

        assert 'if' in directives
        assert 'each' in directives
        assert 'await' in directives
        assert 'key' in directives

    def test_get_script_lang(self):
        """Test extracting script language attribute."""
        mapping = SvelteMapping()

        assert mapping.get_script_lang('lang="ts"') == 'ts'
        assert mapping.get_script_lang('lang="js"') == 'js'
        assert mapping.get_script_lang('') == 'js'  # Default
        assert mapping.get_script_lang("lang='typescript'") == 'typescript'

    def test_is_module_script(self):
        """Test identifying module context scripts."""
        mapping = SvelteMapping()

        assert mapping.is_module_script('context="module"') is True
        assert mapping.is_module_script("context='module'") is True
        assert mapping.is_module_script('lang="ts"') is False
        assert mapping.is_module_script('') is False

    def test_module_and_instance_scripts(self):
        """Test extracting both module and instance script sections."""
        content = '''
<script context="module">
    export const preload = () => {};
</script>

<script>
    let data = [];
</script>
        '''
        mapping = SvelteMapping()
        sections = mapping.extract_sections(content)

        assert len(sections['script']) == 2
        # First script should be module context
        attrs1, content1, _ = sections['script'][0]
        assert 'context="module"' in attrs1
        assert 'export const preload' in content1

        # Second script should be instance
        attrs2, content2, _ = sections['script'][1]
        assert 'context="module"' not in attrs2
        assert 'let data' in content2

    def test_language_property(self):
        """Test that SvelteMapping identifies as SVELTE language."""
        mapping = SvelteMapping()
        assert mapping.language == Language.SVELTE

    def test_inherits_from_typescript(self):
        """Test that SvelteMapping inherits TypeScript capabilities."""
        from chunkhound.parsers.mappings.typescript import TypeScriptMapping

        mapping = SvelteMapping()
        assert isinstance(mapping, TypeScriptMapping)

    def test_empty_markup_not_included(self):
        """Test that empty markup sections are not included."""
        content = '''
<script>
    let x = 1;
</script>

<style>
    div { color: red; }
</style>
        '''
        mapping = SvelteMapping()
        sections = mapping.extract_sections(content)

        # Should have script and style, but no meaningful markup
        assert len(sections['script']) == 1
        assert len(sections['style']) == 1
        # Markup might be empty or contain only whitespace
        if sections['markup']:
            assert sections['markup'][0][1].strip() == ""

    def test_complex_reactive_patterns(self):
        """Test complex reactive declaration patterns."""
        script = '''
// Simple reactive assignment
$: doubled = count * 2;

// Reactive statement with side effects
$: if (count > 10) console.log('high');

// Reactive block
$: {
    const result = expensiveComputation(count);
    console.log(result);
}

// Multiple dependencies
$: sum = a + b + c;
        '''
        mapping = SvelteMapping()
        reactive_declarations = mapping.detect_reactive_declarations(script)

        # Should detect all reactive declarations
        assert len(reactive_declarations) >= 4

    def test_store_autosubscription_patterns(self):
        """Test various store auto-subscription patterns."""
        content = '''
<script>
    import { writable, derived } from 'svelte/store';
    const count = writable(0);
</script>

<div>
    <p>Count: {$count}</p>
    <input bind:value={$name} />
    {#if $isLoggedIn}
        <p>Welcome!</p>
    {/if}
    <button on:click={() => $count.set(0)}>Reset</button>
</div>
        '''
        mapping = SvelteMapping()
        stores = mapping.detect_store_usage(content)

        assert 'count' in stores
        assert 'name' in stores
        assert 'isLoggedIn' in stores

    def test_typescript_script_support(self):
        """Test TypeScript script support."""
        content = '''
<script lang="ts">
    interface User {
        name: string;
        age: number;
    }

    let user: User = {
        name: 'Alice',
        age: 30
    };
</script>

<div>
    <p>Hello {user.name}</p>
</div>
        '''
        mapping = SvelteMapping()
        sections = mapping.extract_sections(content)

        attrs, script_content, _ = sections['script'][0]
        assert mapping.get_script_lang(attrs) == 'ts'
        assert 'interface User' in script_content
        assert 'let user: User' in script_content

    def test_scoped_styles(self):
        """Test extracting scoped style sections."""
        content = '''
<style>
    /* Component-scoped by default */
    p {
        color: purple;
    }
</style>

<style global>
    /* Global styles */
    body {
        margin: 0;
    }
</style>
        '''
        mapping = SvelteMapping()
        sections = mapping.extract_sections(content)

        assert len(sections['style']) == 2
        # First style (scoped by default)
        attrs1, content1, _ = sections['style'][0]
        assert 'color: purple' in content1

        # Second style (global)
        attrs2, content2, _ = sections['style'][1]
        assert 'global' in attrs2
        assert 'margin: 0' in content2


class TestSvelteIntegration:
    """Integration tests for Svelte component parsing."""

    def test_real_world_component(self):
        """Test parsing a realistic Svelte component."""
        content = '''
<script lang="ts">
    import { onMount } from 'svelte';
    import { userStore } from './stores';

    export let title: string;

    let count = 0;
    $: doubled = count * 2;

    onMount(() => {
        console.log('Component mounted');
    });

    function increment() {
        count += 1;
    }
</script>

<main>
    <h1>{title}</h1>
    <p>Count: {count}</p>
    <p>Doubled: {doubled}</p>
    <p>User: {$userStore.name}</p>

    {#if count > 5}
        <p class="warning">Count is getting high!</p>
    {/if}

    {#each [1, 2, 3] as item}
        <span>{item}</span>
    {/each}

    <button on:click={increment}>
        Increment
    </button>
</main>

<style>
    main {
        padding: 1rem;
    }

    .warning {
        color: red;
    }
</style>
        '''
        mapping = SvelteMapping()
        sections = mapping.extract_sections(content)

        # Verify all sections extracted
        assert len(sections['script']) == 1
        assert len(sections['markup']) == 1
        assert len(sections['style']) == 1

        # Verify reactive declarations
        script_content = sections['script'][0][1]
        reactive_decls = mapping.detect_reactive_declarations(script_content)
        assert any('doubled' in decl for decl in reactive_decls)

        # Verify store usage
        stores = mapping.detect_store_usage(content)
        assert 'userStore' in stores

        # Verify directives
        markup_content = sections['markup'][0][1]
        directives = mapping.detect_svelte_directives(markup_content)
        assert 'if' in directives
        assert 'each' in directives

    def test_language_detection_from_extension(self):
        """Test that .svelte extension maps to SVELTE language."""
        from pathlib import Path
        assert Language.from_file_extension(Path('test.svelte')) == Language.SVELTE
        assert Language.from_file_extension(Path('Component.svelte')) == Language.SVELTE
