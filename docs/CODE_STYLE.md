# Code style

How code in this project should read. Follow it for every code change you make, and use it as the checklist when refactoring older code to match. When in doubt, match the style of the surrounding code that already follows this guide.

## Goal

Write code that a junior to mid-level developer can read, reason about and change with confidence.

## Structure

- **Single responsibility.** Each function, class, method and file has one reason to change, conceptually. That doesn't mean one-line functions.
- **DRY.** When the same logic appears in more than one place, extract it into a reusable function.
- **New files are fine** for helpers or utility classes, as long as each file's contents fit together conceptually.
- **Design patterns are welcome** when they clearly improve the code, even if they add some complexity. Every pattern gets a comment explaining which pattern it is, plus a short note on how to use it and how to extend it (for example, "To support a new X, add a class that implements Y and register it in Z").
- **Use interfaces to decouple** parts of the code that don't need to know about each other.
- **Avoid most `if`/`else if` chains.** Prefer a lookup table, a list of strategies or a list of checks that you loop over. Keep an if/else when it really is the clearest option.

  ```ts
  // Avoid
  function describeStatus(status: string): string {
    if (status === "open") {
      return "Waiting for review";
    } else if (status === "closed") {
      return "No longer accepting changes";
    } else if (status === "merged") {
      return "Done";
    }
    return "Unknown";
  }

  // Prefer
  const STATUS_DESCRIPTIONS: Record<string, string> = {
    open: "Waiting for review",
    closed: "No longer accepting changes",
    merged: "Done",
  };

  function describeStatus(status: string): string {
    return STATUS_DESCRIPTIONS[status] ?? "Unknown";
  }
  ```

- **Prefer configurable code to inline values.** Give numbers and strings a named constant or setting instead of writing them inline (`MAX_RETRIES`, `PAGE_LOAD_TIMEOUT_MS`), and group related settings together in one place.

## Size and shape

- **Functions and methods stay under about 20 lines.** This is a guide, not a hard limit, so go over it when that's genuinely clearer.
- **A line does two or three things at most.** Assign intermediate results to well-named variables instead of nesting calls. For example, don't map and join inside a template string inside a log call:

  ```ts
  // Avoid
  console.log(`\n${title}:\n${rows.map((r) => `  - ${r}`).join("\n")}`);

  // Prefer
  const bulletLines = rows.map((row) => `  - ${row}`);
  const list = bulletLines.join("\n");
  console.log(`\n${title}:\n${list}`);
  ```

- **Separate "phrases" with blank lines.** A phrase is like a paragraph: a few lines that do one related thing. Put a blank line between one idea and the next.
- **Format with the project's formatter** using its default settings (in this Python project, Ruff: `uv run ruff format`). Don't hand-format around it.

## Names

- **Descriptive names beat short ones**, for files, functions, classes, methods and variables.
- **Functions are named as actions**, starting with a verb that says what they do, even when they're small: `joinErrors`, not `bothErrors`; `buildReport`, not `report`.
- **Booleans read as conditions**: `isCondition` or `hasCondition` (`isDryRun`, `isSubmitted`, `hasLoaded`). That includes functions that return booleans (`isValidEmail`, `hasPermission`).

## Making changes safely

- **Tests pass before and after.** Run the test suite before you start, and again whenever the code should work partway through a refactor. Run the type checker and linter too, if the project has them.
- **Make sure the tests actually ran.** Watch for tests that pass without testing anything, such as ones that skip silently when a dependency (a browser, a database, a service) isn't available. If a test was skipped, say so instead of reporting it as passing.
- **Refactors don't change behavior.** Keep refactors and behavior changes in separate commits or pull requests.
- **Stay inside the area you're working on.** Be careful changing code in other parts of the project, especially fragile or critical areas. Small mechanical changes there, like updating an import, are fine.
- **Respect code with special runtime constraints.** Some code can't use normal imports or helpers, such as functions serialized and run in another context (a browser page, a worker, a sandbox). Keep those self-contained, leave a comment explaining the constraint, and test them in the same runtime they really run in.
