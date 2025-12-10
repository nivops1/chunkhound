---
description: Critical code review before committing changes
argument-hint: [description of changes]
---

You are an expert code reviewer. Analyze the current changeset and provide a critical review.

The changes in the working tree were meant to: $ARGUMENTS

Think step-by-step through each aspect below, focusing solely on the changes in the working tree.

1. **Architecture & Design**
   - Verify conformance to project architecture
   - Check module responsibilities are respected
   - Ensure changes align with the original intent

2. **Code Quality**
   - Code must be self-explanatory and readable
   - Style must match surrounding code patterns
   - Changes must be minimal - nothing unneeded
   - Follow KISS principle

3. **Maintainability**
   - Optimize for future LLM agents working on the codebase
   - Ensure intent is clear and unambiguous
   - Verify comments and docs remain in sync with code

4. **User Experience**
   - Identify areas where extra effort would significantly improve UX
   - Balance simplicity with meaningful enhancements

Review the changes critically. Focus on issues that matter.
DO NOT EDIT ANYTHING - only review.