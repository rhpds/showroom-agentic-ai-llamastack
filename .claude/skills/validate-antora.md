---
description: Validate Antora navigation and content files for completeness and correctness
---

# Antora Content Validator

## Overview

This skill validates Antora documentation structure by checking:
1. Navigation file (nav.adoc) syntax and references
2. Existence of all referenced content files
3. Content file structure and internal cross-references
4. Broken links and missing assets

## Validation Process

### Step 1: Parse Navigation File

Read and parse `content/modules/ROOT/nav.adoc`:
- Extract all `xref:` references
- Extract default index references from comments
- Identify the navigation structure and hierarchy
- Check for proper AsciiDoc syntax in navigation entries

### Step 2: Validate File Existence

For each referenced file in nav.adoc:
- Verify the file exists in `content/modules/ROOT/pages/`
- Check file naming conventions (kebab-case, .adoc extension)
- Report any missing files
- Report any files in pages/ that are not referenced in nav.adoc (orphaned files)

### Step 3: Validate Content Files

For each content file in pages/:
- **Title validation**: Ensure file has a level-1 heading (=)
- **Cross-reference validation**: Extract all `xref:` links and verify targets exist
- **Include directives**: Check all `include::` directives point to existing files
- **Image references**: Validate `image::` and `image:` directives (if applicable)
- **Anchor validation**: Check for duplicate IDs and broken anchor references
- **Mermaid diagrams**: Verify mermaid blocks have proper delimiters and syntax
- **Code blocks**: Check that source blocks are properly closed
- **Attribute references**: Identify undefined attribute references

### Step 4: Content Quality Checks

- Check for common AsciiDoc syntax errors:
  - Unclosed blocks
  - Malformed list structures
  - Invalid table syntax
  - Missing blank lines before blocks
- Verify consistent heading hierarchy (no skipped levels)
- Check for empty files or files with only a title

### Step 5: Generate Validation Report

Create a comprehensive report with:
- **Summary**: Total files checked, issues found, overall status
- **Navigation Issues**: Missing references, syntax errors
- **Missing Files**: Files referenced but not found
- **Orphaned Files**: Files not referenced in navigation
- **Content Issues**: By file, listing all validation errors
- **Recommendations**: Suggested fixes for common issues

## Implementation Guidelines

1. **Use appropriate tools**:
   - Read tool for parsing nav.adoc and content files
   - Grep tool for finding patterns (xref, include, image directives)
   - Glob tool for finding all .adoc files

2. **Error handling**:
   - Gracefully handle missing files
   - Report parsing errors without stopping validation
   - Provide clear, actionable error messages

3. **Output format**:
   - Use markdown formatting for the report
   - Include file paths with line numbers when possible
   - Use severity levels: ERROR, WARNING, INFO
   - Provide examples of correct syntax for errors

4. **Performance**:
   - Read each file only once
   - Use parallel reads when possible
   - Cache file existence checks

## Expected Output Format

```markdown
# Antora Content Validation Report

## Summary
- Total files: X
- Files validated: X
- Issues found: X
- Status: ✓ PASS / ✗ FAIL

## Navigation Validation
### nav.adoc (content/modules/ROOT/nav.adoc)
- ✓ Syntax valid
- ✓ All references resolved
- Referenced files: X

## File Existence Check
### Missing Files (ERROR)
- [ ] `example.adoc` - Referenced in nav.adoc:5

### Orphaned Files (WARNING)
- [ ] `unreferenced.adoc` - Exists but not in navigation

## Content Validation
### index.adoc
- ✓ Has level-1 heading
- ✓ No broken cross-references
- ✓ Syntax valid

### 02-llamastack.adoc
- ✓ Has level-1 heading
- ✗ ERROR: Broken xref at line 42: `xref:nonexistent.adoc`
- ⚠ WARNING: Image not found at line 88: `image::missing.png`
- ✓ Mermaid diagram syntax valid (2 diagrams)

## Recommendations
1. Fix broken xref in 02-llamastack.adoc:42
2. Add missing image or update reference in 02-llamastack.adoc:88
3. Consider adding unreferenced.adoc to navigation or removing it

## Details
[Detailed error listings with line numbers and context]
```

## Usage Examples

**Basic validation:**
```
User: Validate the Antora content
Agent: [Runs full validation and provides report]
```

**Focused validation:**
```
User: Check if all navigation links work
Agent: [Validates nav.adoc and checks file existence only]
```

**After content updates:**
```
User: I just added a new page, validate the docs
Agent: [Runs validation and reports any issues with the new content]
```

## Skill Metadata

- **Trigger patterns**:
  - "validate antora"
  - "check antora content"
  - "validate docs"
  - "check navigation"
  - "antora validation"

- **Project scope**: Antora-based documentation projects
- **Required files**: content/modules/ROOT/nav.adoc, content/modules/ROOT/pages/
- **Dependencies**: None (uses built-in Claude Code tools)

## Advanced Features

### Auto-fix Mode (Optional)
When user requests it, the skill can:
- Add missing files to navigation
- Remove broken references
- Create stub files for missing pages
- Fix common syntax errors

### Integration Points
- Can be used as a pre-commit hook
- Can be integrated into CI/CD pipelines
- Can generate GitHub-flavored markdown for issues

## Notes

- This skill follows Antora's standard module structure (ROOT/pages, ROOT/nav.adoc)
- Can be extended to validate attachments, examples, and partials directories
- Supports standard AsciiDoc syntax and Antora-specific extensions
- Does not validate external links (URLs) by default to avoid rate limiting
