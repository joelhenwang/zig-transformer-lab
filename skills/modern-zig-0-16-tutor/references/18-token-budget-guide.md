# 18 — Token budget guide for consumers of this skill

This file tells the AI consumer how to load this skill **progressively** so
that a typical 200 k context window stays healthy. Ignoring this guide leads
to the classic failure mode where the skill itself eats 40 % of context and
the user's code can't fit.

## Per-turn budget

Reserve for a typical turn:

| Slot | Budget |
|---|---|
| System prompt + tool metadata | ~8 k |
| User's pasted code / errors | up to ~30 k |
| Tool output during the turn | up to ~20 k |
| Response headroom | ~8–16 k |
| **Skill content** | ~30–40 k max |

Your target for skill content:

- **Typical turn**: ~6–12 k tokens (SKILL.md + 1 reference + maybe 1 recipe)
- **Heavy audit turn**: ~20–25 k tokens (SKILL.md + 2 refs + user code)

## Loading rules

1. **Always load `SKILL.md`.** It is ~3 k tokens. You have no choice.
2. **Load at most 3 reference files per turn**, unless the user explicitly
   asks for a broad audit.
3. **Prefer recipes over references for "how do I" questions.** Recipes are
   ~½ the tokens and they already link to the relevant reference.
4. **Defer reference loading** when the user pastes more than 5 k tokens of
   code. Read the code first.
5. **For migration audits**, load `references/01` + `references/14`. Do
   **not** also load 08/09 — their migrations are already in 14.
6. **For code review**, load `references/15` + `references/01`. Refer to
   other references by path, don't load them.
7. **For CUDA**, load `references/17` + `references/09`. Do not also load
   `references/16` unless the user is building a tensor API.
8. **For I/O questions**, load `references/07`. Do not load 08 unless the
   user's issue involves `build.zig`.

## Approximate file sizes (in tokens)

| File | Tokens | Notes |
|---|---|---|
| `SKILL.md` | ~3.0 k | Always loaded |
| `references/00-version-policy.md` | ~1.2 k | Cheap; load on version questions |
| `references/01-zig-0-16-critical-changes.md` | ~4.0 k | Anchor; load often |
| `references/02-language-basics.md` | ~2.0 k | |
| `references/03-types-pointers-slices.md` | ~2.2 k | |
| `references/04-errors-optionals-defer.md` | ~2.0 k | |
| `references/05-memory-allocators.md` | ~2.2 k | |
| `references/06-containers-0-16.md` | ~2.0 k | |
| `references/07-io-0-16.md` | ~2.5 k | |
| `references/08-build-system-0-16.md` | ~2.8 k | |
| `references/09-c-interop-0-16.md` | ~2.8 k | |
| `references/10-testing-debugging.md` | ~2.2 k | |
| `references/11-comptime-metaprogramming.md` | ~2.4 k | |
| `references/12-formatting-logging.md` | ~2.0 k | |
| `references/13-style-guide.md` | ~2.0 k | |
| `references/14-migration-from-0-13-to-0-16.md` | ~3.4 k | |
| `references/15-code-review-checklist.md` | ~3.0 k | |
| `references/16-zig-for-ml-runtime-projects.md` | ~2.6 k | |
| `references/17-zig-cuda-interop-notes.md` | ~2.8 k | |
| `references/18-token-budget-guide.md` | ~1.2 k | This file |
| Each recipe file | ~1.0 k | |
| Each template `.zig` | ~0.8–1.4 k | |
| Each example `.zig` | ~0.7 k | |

## Anti-patterns to avoid

- **Loading every reference "just in case".** If you load more than 3
  references and the task doesn't need them, you are burning user budget.
- **Quoting large sections of a reference in the response.** Prefer a path
  reference: "see `references/07-io-0-16.md#flushing`".
- **Re-quoting `SKILL.md` content.** The consumer already has it.
- **Loading a reference, then answering from memory.** If you loaded it,
  quote the exact snippet or definition.
- **Loading all of `references/`** in an auto-completion loop. Be explicit
  about which file you need and why.

## When to break these rules

- User says "do a full audit of my project" → load `01`, `14`, `15` together.
- User is a first-time learner, overwhelmed by Zig → load `02`, `04`, `05`,
  `06`, `10` across turns as needed, not all at once.
- User pastes a build error with unfamiliar text → load `01` first; if the
  root cause is build-system-related, add `08`.

## Self-check before loading

Ask yourself:

1. Does the user's question actually require this reference, or can I
   answer from `SKILL.md`'s summary + a path link?
2. Am I about to load two references that mostly overlap (e.g. 01 and 14)?
   Pick one.
3. Will my response re-quote half of the reference I loaded? If yes, the
   user would have been served as well by a path link.

If all three answers are "no", proceed with the load.

<!-- ~1.3k tokens -->
