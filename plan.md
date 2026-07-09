# RAG Against the Machine — Project Plan

> A guided roadmap, not a solution. Code blocks are boilerplate/scaffolding only —
> the retrieval, chunking, and prompt-design logic is intentionally left for you to
> write, since that's the actual point of the exercise (and what you'll be asked to
> defend during evaluation).

---

## 0. Read this first: what the moulinette actually tests

Before writing anything, internalize the shape of the grading pipeline described in
§V.6 of the subject. It is **not** "run one command and get an answer." It's a
5-stage pipeline where each stage reads/writes files on disk, and can be invoked
independently:

```
index  →  search / search_dataset  →  evaluate  →  answer / answer_dataset
```

The evaluator will run `index` once, then call `search_dataset` on a hidden
question set, then run `evaluate` (their own `moulinette` module, not yours) against
your search output. This means:

- Your `search`/`search_dataset` output must be **valid, self-contained JSON** that
  matches `StudentSearchResults` exactly — no extra required fields, no missing ones.
- Your index must be **persisted to disk** and reloadable without re-indexing,
  because `search` will likely be invoked as a fresh process, not continuing from
  `index`'s memory.
- Character offsets are the actual "answer" your retrieval is graded on — get this
  exactly right before optimizing anything else.

Keep this pipeline picture in your head the whole time; it should guide how you
structure state (files, not global variables) far more than typical script design.

---

## 1. Project skeleton

Set this up yourself with `uv init` — it's pure boilerplate and worth doing by hand
once so you understand what `uv` generates.

```
your-repo/
├── pyproject.toml          # uv-managed
├── uv.lock
├── Makefile                # install / run / debug / clean / lint / lint-strict
├── .gitignore               # exclude data/, __pycache__, .mypy_cache, .venv, etc.
├── README.md
├── src/
│   └── student/
│       ├── __init__.py
│       ├── __main__.py     # entry point: `python -m student` -> fire.Fire(CLI)
│       ├── cli.py          # Fire-exposed class, the 6 commands
│       ├── models.py       # pydantic models (given + your extensions)
│       ├── chunking.py     # ← interesting: code + text chunking strategies
│       ├── indexing.py     # build/save/load BM25 (or TF-IDF) index
│       ├── retrieval.py    # ← interesting: query → ranked chunks
│       ├── generation.py   # LLM prompt construction + inference wrapper
│       └── evaluation.py   # recall@k computation
└── data/                    # gitignored; generated at runtime by `index` etc.
    ├── raw/                 # extracted vLLM repo goes here (not committed)
    ├── processed/           # index + chunks
    └── output/               # search_results, search_results_and_answer
```

### Why `src/student/` and not a flat file?

The subject invokes your CLI as `python -m student <command> ...` (see §V.6.2–V.6.7
transcripts). That means your package must be importable as `student`, and
`__main__.py` must exist so `-m student` resolves. If you use a `src/` layout,
make sure `pyproject.toml` points `tool.uv` / build backend at `src/student`, or
just install in editable mode (`uv pip install -e .`) and verify `python -m student
--help` works before writing any real logic. This trips people up more often than
you'd expect — test it on day one, not day five.

### Makefile targets — do these first, they're cheap wins

| Target | What it must do |
|---|---|
| `install` | `uv sync` (or pip/pipx equivalent) |
| `run` | `uv run python -m student <default behavior>` — decide what "running" means with no args, e.g. print help |
| `debug` | same as `run` but drops into `pdb` on exception, or runs with `python -m pdb` |
| `clean` | remove `__pycache__`, `.mypy_cache`, `.pytest_cache`, build artifacts |
| `lint` | `flake8 .` and `mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs` |
| `lint-strict` (optional) | `flake8 .` and `mypy . --strict` |

Get `lint` passing on the empty skeleton before you write a single line of chunking
logic — much easier to keep clean than to retrofit type hints onto 400 lines later.

---

## 2. Data models (`models.py`)

Transcribe the pydantic models from §V.7 of the subject **exactly** — field names
and types matter because the evaluator's `moulinette.evaluate_student_search_results`
will parse your JSON output directly against these schemas. A typo like
`retrieved_source` instead of `retrieved_sources` will silently fail validation.

The six models you're given:

- `MinimalSource` — `file_path: str`, `first_character_index: int`, `last_character_index: int`
- `UnansweredQuestion` — `question_id: str` (UUID default), `question: str`
- `AnsweredQuestion(UnansweredQuestion)` — adds `sources: List[MinimalSource]`, `answer: str`
- `RagDataset` — `rag_questions: List[AnsweredQuestion | UnansweredQuestion]`
- `MinimalSearchResults` — `question_id`, `question`, `retrieved_sources: List[MinimalSource]`
- `MinimalAnswer(MinimalSearchResults)` — adds `answer: str`
- `StudentSearchResults` — `search_results: List[MinimalSearchResults]`, `k: int`
- `StudentSearchResultsAndAnswer(StudentSearchResults)` — overrides `search_results: List[MinimalAnswer]`

### Your own addition: a `Chunk` model

You'll need something to represent an indexed chunk before it becomes a
`MinimalSource`. Two reasonable designs — pick one and justify it in your README:

1. **Fat chunk model**: `Chunk(BaseModel)` with `chunk_id`, `file_path`,
   `first_character_index`, `last_character_index`, `content`, `chunk_type`
   (`"code"` / `"text"`), maybe `token_count`. Store all of these in your persisted
   index so `search` never needs to re-read source files from disk.
2. **Thin approach**: reuse `MinimalSource` as the identity key, keep chunk text in
   a separate lookup keyed by `(file_path, first_character_index,
   last_character_index)`, loaded alongside the BM25 index.

Option 1 is simpler to reason about and debug; option 2 is more "correct" relative
to not duplicating the given models. Either is fine — the important thing is that
**your persisted index must let `search` reconstruct exact file_path + offsets
without re-reading or re-chunking the repo**, since re-chunking would violate the
"index once, search many times" performance contract (cold-start ≤60s, warm 1000
questions ≤90s).

---

## 3. Chunking strategy (`chunking.py`) — write this yourself

This is the single most consequential design decision in the whole project. Bad
chunking caps your recall@k no matter how good your retrieval math is, because you
can't retrieve information that got split across a boundary or buried in a
1200-character blob of irrelevant code.

### Hard constraints from the subject

- Max chunk size: **2000 characters**, and this must be a **CLI-configurable
  argument** (`--max_chunk_size`), not a hardcoded constant.
- You need **two distinct strategies**: one for Python code, one for text/Markdown.
- Every chunk must carry `first_character_index` / `last_character_index` as
  offsets **into the original file's raw text**, not offsets relative to some
  intermediate representation. Get this wrong and every downstream recall number is
  meaningless, even if your retrieval logic is perfect.

### Strategy A: naive baseline (build this first)

Fixed-size sliding window over the raw file text, respecting `max_chunk_size`,
optionally with a small overlap (e.g. 100–200 characters) so information straddling
a boundary isn't lost entirely from either neighboring chunk. This should take you
maybe 30 minutes to write and is enough to get the *whole pipeline* working
end-to-end — indexing, searching, evaluating — before you touch anything smarter.

**Why start here even though it's obviously not the "best" chunking**: the "don't
panic" tip in the subject (§V.3) is explicit about this — measure your error with
the simplest approach first. If you jump straight to AST-aware chunking and
something's broken, you won't know whether the bug is in chunking, indexing,
retrieval, or your offset math. Isolate variables.

### Strategy B: text/Markdown chunking (the "interesting" text one)

Once the naive baseline works end-to-end, upgrade the text/Markdown path. Markdown
has structure you should exploit:

- Split on headers (`#`, `##`, `###`) first — treat each section as a candidate
  chunk, since questions about documentation usually map cleanly onto one section
  (e.g. "How to configure OpenAI server?" → the section literally titled that).
- If a section exceeds `max_chunk_size`, recursively split it further (by
  paragraph, then by sentence/fixed-size as a last resort) rather than
  truncating — you don't want to silently drop the back half of a long section.
- If a section is much smaller than `max_chunk_size`, consider whether to leave it
  standalone (preserves semantic coherence, but may hurt BM25's ability to compute
  meaningful term frequencies on very short documents) or merge adjacent small
  sections. This is a real tradeoff — try both and look at your recall numbers.
- Decide how to handle code blocks embedded inside Markdown (fenced ` ``` ` blocks)
  — do they get chunked as part of the surrounding text, or extracted and chunked
  as pseudo-code? Either is defensible; document your choice.

### Strategy C: Python code chunking (the "interesting" code one)

This is where the project rewards actually understanding the `ast` module rather
than treating code as generic text.

- Parse each `.py` file with `ast.parse`, then walk top-level nodes
  (`ast.FunctionDef`, `ast.AsyncFunctionDef`, `ast.ClassDef`) to find natural
  chunk boundaries — a function or class body is a coherent unit of meaning in a
  way that "lines 1400–1600" is not.
- Each `ast` node gives you `.lineno` and `.end_lineno` — you'll need to convert
  these to **character offsets** in the raw source (build a line→char-offset
  index once per file, e.g. via cumulative `len()` of each line including
  newlines, and reuse it for every node in that file rather than recomputing per
  chunk).
- Decide what to do with:
  - **Module-level code** outside any function/class (imports, constants, `if
    __name__ == "__main__"` blocks) — its own chunk, or merged into the nearest
    function?
  - **Class methods**: chunk the whole class as one unit, or split large classes
    into per-method chunks (with the class docstring/signature repeated as
    context in each)? Large classes (vLLM has some big ones, e.g. in
    `vllm/model_executor/`) will blow past 2000 chars easily if kept whole.
  - **Functions bigger than 2000 chars**: fall back to sub-splitting (e.g. by
    top-level statements inside the function body), same recursive-split idea as
    the Markdown case.
  - **Docstrings and decorators**: keep them attached to their function/class —
    they're often exactly the natural-language text that makes a code chunk
    retrievable by a natural-language query in the first place.
- **Tokenization detail worth experimenting with**: BM25 matches on tokens. A
  query like *"What method needs to be overridden in BaseProcessingInfo..."*
  needs to match against a chunk containing `get_supported_mm_limits`. Consider
  splitting identifiers on `snake_case` / `camelCase` boundaries into subtokens
  (`get`, `supported`, `mm`, `limits`) at index time — this is often the single
  biggest lever for boosting code recall, bigger than chunk-boundary tuning.

### What to explicitly punt on

Don't try to handle every Python edge case (walrus operators inside comprehensions,
weirdly nested closures, etc.) — wrap the `ast.parse` call in a try/except and fall
back to Strategy A for any file that fails to parse (some files in a large repo
may have syntax quirks, encoding issues, or be Python-version-specific syntax your
parser chokes on). Log a warning, don't crash — this is also just good practice
per the "handle exceptions gracefully" rule in §III.1/§IV.1.

---

## 4. Indexing (`indexing.py`)

### What "index all the files you judge useful" means in practice

The vLLM repo (as unzipped) contains ~1900 `.py` files and ~180 `.md` files across
areas like `vllm/`, `docs/`, `tests/`, `.github/`, `benchmarks/`, `examples/`. You
need to make and **justify in your README** a filtering decision:

- Almost certainly include: `vllm/**/*.py` (the actual library code) and
  `docs/**/*.md` (user-facing documentation) — these map directly to the kind of
  questions in `dataset_docs_public.json` / `dataset_code_public.json`.
- Probably exclude: `.github/` workflow YAML, `tests/` (unless questions target
  test code specifically — check the sample datasets to see), generated/vendored
  code, binary assets.
- Look at a sample of the provided `dataset_code_public.json` /
  `dataset_docs_public.json` (in `AnsweredQuestions/`) **before** finalizing your
  filter — the ground-truth `sources` field tells you exactly which directories
  the real questions target. Don't guess; check.

### Build steps

1. Walk the filtered file list.
2. For each file, dispatch to Strategy B or C from `chunking.py` based on
   extension.
3. Collect all chunks into your chosen chunk model (see §2).
4. Build a BM25 index over chunk contents (`bm25s` is the recommended package —
   it's fast and designed for exactly this; TF-IDF via `sklearn` is a fine simpler
   alternative if you want to start there and swap later).
5. Persist to `data/processed/`: the BM25 index artifacts plus a serialized chunk
   list (so `search` can map BM25 result indices back to `file_path` +
   offsets + text without re-reading source files).

### Performance constraint: 5-minute indexing cap

For ~1900+180 files this should not be a binding constraint if your chunking is
reasonably efficient (BM25 indexing itself is fast — seconds, not minutes, even
for tens of thousands of chunks). If you're close to the 5-minute limit, the
bottleneck is almost certainly your chunking step (e.g. re-parsing the same AST
multiple times, or doing expensive per-character regex operations) — profile
before optimizing blindly.

### One index or several?

The subject's tip about "linked to the different chunking strategies, you can
create different indexes for the different types of files" (§V.8) is a hint worth
taking. A combined index is simpler; separate code/docs indexes let you:

- Tune BM25 hyperparameters (`k1`, `b`) independently per content type.
- Potentially route a query to the more relevant index type first (e.g. detect
  "how do I configure..." as doc-flavored vs. "what does `get_supported_mm_limits`
  do" as code-flavored) — though simple heuristics here can easily hurt more than
  help; measure before committing to routing logic.

Try the combined index first (simpler, one less thing to get wrong), measure
recall separately for docs vs. code questions (which the evaluator already does —
see §VI.1.2), and only split into separate indexes if you have concrete evidence
one content type is dragging the other down.

---

## 5. Retrieval (`retrieval.py`) — write this yourself

### Baseline

1. Tokenize the incoming query with **exactly the same tokenizer/preprocessing**
   you used at index time (lowercasing, stopword handling, identifier-splitting if
   you did that in chunking) — mismatched tokenization between index-time and
   query-time is a classic silent bug that tanks recall for no obvious reason.
2. Score all chunks against the query via BM25.
3. Take top-k, convert each to a `MinimalSource` (file_path + offsets).

### Where the real gains are (in rough order of expected impact)

1. **Identifier subtokenization for code** (mentioned in §3) — often the biggest
   single win, because natural-language queries rarely use exact `snake_case`
   function names as substrings.
2. **Chunk boundary quality** — AST-aware code chunks and header-aware doc chunks
   (§3) beat fixed windows because they keep semantically related text together,
   which is what lets a single chunk fully "cover" a ground-truth answer.
3. **k tuning** — recall@k is monotonically non-decreasing in k, but your CLI
   default `k` and what you pass during your own evaluation runs should be chosen
   deliberately, not left at whatever felt convenient.
4. **Score thresholding / result count** — should `search` always return exactly
   k results, or fewer if scores drop below some relevance floor? The subject
   doesn't mandate a threshold, but returning obviously irrelevant low-score
   chunks as padding can hurt if it ever interacts with your context-window
   truncation logic in `answer`.

### Debugging workflow (use this, it's faster than guessing)

Run `evaluate` on the public `AnsweredQuestions` datasets, then use the `jq`
inspection pattern from §V.6.8 to pull out specific mispredicted questions —
compare `expected` sources against what you actually retrieved for that
`question_id`. Look for patterns: are misses concentrated in one file type, one
directory, very short ground-truth spans, or very long ones? Let *evidence* from
these misses drive your next chunking/retrieval iteration rather than tweaking
blindly.

---

## 6. Answer generation (`generation.py`)

This part is comparatively mechanical — the interesting work is in the prompt
design, not the plumbing.

### Plumbing (straightforward)

- Load `Qwen/Qwen3-0.6B` via `transformers` (`AutoModelForCausalLM` +
  `AutoTokenizer`).
- Build a prompt combining: a system/instruction preamble, the retrieved chunk
  texts (with enough metadata — e.g. file path — that the model can cite sources),
  and the question.
- Truncate context using the **tokenizer's** token count, not raw character
  count, to respect the token-limit constraint from §V.2.3 — character count is a
  poor proxy for token count, especially for code with lots of punctuation/symbols.
- Run generation, extract the text, wrap it in a `MinimalAnswer`.

### The actually interesting part: prompt design

The subject defines four required qualities for a good answer (§V.2.3) — design
your prompt (and possibly post-processing) around each one explicitly:

- **Self-contained**: the answer must make sense to someone who never saw the
  question. This usually means instructing the model to restate necessary context
  rather than starting with "Yes," or "You need to...", referring back to a "the
  question asks" that isn't visible to the reader.
- **Source-grounded**: the answer should reference which file(s)/chunk(s) it
  drew from. Consider whether this citation happens inside the generated text
  itself, or whether you rely on the `retrieved_sources` field in your output
  JSON to carry that information — the subject doesn't fully specify which, so
  pick one and be able to justify it.
- **Faithful (no hallucination)**: with only a 0.6B model, hallucination risk is
  real. Explicit instruction ("only use information present in the context below;
  if the context doesn't contain the answer, say so") plus keeping retrieved
  context tight and relevant (good retrieval reduces hallucination pressure far
  more than prompt wording alone) both matter here.
- **Relevant**: directly answers what was asked, not a general summary of the
  retrieved chunks. Watch for small models drifting into restating chunk content
  verbatim instead of synthesizing a targeted answer — you may need few-shot
  examples in the prompt if zero-shot instructions aren't enough.

Iterate on this by hand against a handful of real questions from the public
dataset before running it over the full 100-question set — prompt engineering is
much faster to debug on 5 examples than 100.

---

## 7. Evaluation (`evaluation.py`)

This module is small (maybe 20–40 lines of real logic) but has real edge cases —
worth writing yourself rather than outsourcing, since it's short and the overlap
math is exactly the kind of thing that's easy to get subtly wrong.

### The definition, precisely (§VI.1.1)

For a given question with ground-truth sources `G = {g_1, ..., g_n}` and your
retrieved sources `R = {r_1, ..., r_m}`:

- A ground-truth source `g_i` counts as **"found"** if there exists at least one
  retrieved source `r_j` such that they're on the **same file_path** and their
  character ranges overlap by at least 5% — but 5% of *what*, exactly, is worth
  thinking through carefully (5% of the ground-truth range's length is the most
  natural reading, matching "overlap between the retrieved source and any correct
  source").
- Per-question score = `number_found / total_number_of_correct_sources` (i.e.
  `|found ground-truth sources| / n`).
- Recall@k = mean of per-question scores across the whole dataset, for a given k.

### Edge cases to explicitly handle

- Zero ground-truth sources for a question — avoid a divide-by-zero; decide
  whether that question is excluded from the mean or scored as trivially 1.0 (or
  0.0 — think about which is defensible, and document it).
- Ground truth and retrieved chunk on the same file but zero-length or
  degenerate ranges (`first_character_index == last_character_index`).
- Multiple retrieved chunks overlapping the *same* ground-truth source — should
  this double-count? No: a ground-truth source is found or not found, not found
  multiple times.
- k larger than the number of results actually returned (e.g. you returned fewer
  than k due to a relevance floor) — recall@k should still be computable, just
  based on however many you actually returned.

### Testing your own evaluator

Before trusting your recall numbers, hand-construct 2–3 tiny synthetic examples
with known overlap percentages (e.g. exactly 5% overlap, exactly 4.9%, complete
overlap, zero overlap) and assert your function returns the expected found/not-found
verdict on each. This is cheap insurance against silently wrong recall numbers
driving all your later tuning decisions.

---

## 8. CLI (`cli.py`)

Thin glue layer — write this fast, it's not where the interesting decisions live.
A single class exposed via `fire.Fire`, with roughly this shape:

```python
class CLI:
    def index(self, repo_path: str = "data/raw", max_chunk_size: int = 2000) -> None:
        """Index the repository."""
        ...

    def search(self, query: str, k: int = 10) -> None:
        """Search for a single query, print results."""
        ...

    def search_dataset(self, dataset_path: str, k: int = 10,
                        save_directory: str = "data/output/search_results") -> None:
        """Batch search over a JSON dataset, save StudentSearchResults."""
        ...

    def answer(self, question: str, k: int = 10) -> None:
        """Answer a single question end-to-end (search + generate)."""
        ...

    def answer_dataset(self, student_search_results_path: str,
                        save_directory: str = "data/output/search_results_and_answer") -> None:
        """Generate answers for a pre-computed search-results file."""
        ...

    def evaluate(self, student_answer_path: str, dataset_path: str, k: int = 10,
                 max_context_length: int = 2000) -> None:
        """Compute recall@k against ground truth (your own copy, for local dev)."""
        ...
```

A few things worth getting right here even though it's "just glue":

- **Every method needs a docstring and full type hints** — this is graded
  (§III.1/§IV.1), and Fire will surface docstrings in `--help` output for free,
  so it's not wasted effort.
- **Progress bars** (`tqdm`) belong on `index` (looping over files) and
  `search_dataset`/`answer_dataset` (looping over questions) — anywhere the
  subject calls "long-running operations."
- **Graceful degenerate-input handling** is explicitly tested (§V.2.5): what
  happens with `k=0`? Empty query string? A `dataset_path` pointing at a
  nonexistent file? Malformed JSON in the dataset? None of these should raise an
  unhandled traceback — catch, print a clear error message, exit non-zero.
- Don't put real logic in this file — if a CLI method starts growing past ~15
  lines, that logic almost certainly belongs in `indexing.py`, `retrieval.py`, etc.
  Keeping `cli.py` thin makes it much easier to unit-test the actual logic without
  going through Fire/argv parsing.

---

## 9. Suggested build order

This sequencing is designed to surface bugs early and keep each step
independently testable — resist the urge to jump ahead to "the interesting part"
before the previous stage is verified correct.

1. **Skeleton + models + Makefile + CLI stub.** Every method just prints its
   arguments. Get `make install`, `make lint`, `python -m student --help` all
   working. Zero real logic yet.
2. **Naive fixed-size chunking, both file types, no AST/header smarts.** Focus
   entirely on getting `first_character_index`/`last_character_index` exactly
   right — write a tiny standalone test that chunks a known string and asserts
   `original_text[chunk.first:chunk.last] == chunk.content` (or however you've
   defined the boundary convention — inclusive/exclusive end matters, pick one
   and be consistent everywhere).
3. **Indexing + naive BM25 search, wired end-to-end.** Run `index` then `search`
   on one hand-picked query, sanity-check the results make sense by eye.
4. **Your own `evaluate`, tested against synthetic examples (§7).** Then run it
   for real against the public `AnsweredQuestions` datasets to get your first
   honest recall@k baseline — even if it's low. This number is your yardstick for
   every change from here on.
5. **Iterate on chunking/tokenization**, using the `jq` inspection workflow (§5)
   to find *why* specific questions fail, not by guessing. Alternate between docs
   and code recall — they'll likely need different fixes.
6. **Wire in Qwen3-0.6B generation** once retrieval recall is in a good place —
   there's little point prompt-engineering against a retrieval system that isn't
   finding the right chunks yet.
7. **Harden**: degenerate CLI inputs, exception handling everywhere, `.gitignore`,
   `flake8`/`mypy --strict` clean.
8. **README.md**, written last so it accurately reflects what you actually built
   (see §10 below for required structure).
9. **Bonus features**, only after every mandatory-part threshold is met and
   comfortably re-verified.

---

## 10. README requirements checklist

Required by §VII of the subject — build this incrementally as you make design
decisions (not all at the end from memory) so nothing gets forgotten:

- [ ] First line, italicized: *This project has been created as part of the 42
  curriculum by \<login1\>[, \<login2\>[, \<login3\>[...]]].*
- [ ] **Description** — goal + brief overview
- [ ] **Instructions** — compile/install/run steps (should map 1:1 to your
  `Makefile` targets)
- [ ] **Resources** — references you actually used (BM25 papers/docs, `ast`
  module docs, whatever informed your chunking design) **and** an honest account
  of how AI was used: which tasks, which parts of the project, what you changed
  or verified afterward. This is not a formality — §II of the subject makes clear
  you can fail your defense if you can't explain AI-assisted code.
- [ ] **System architecture** — the pipeline diagram from §0 above, in your own
  words, describing your actual modules
- [ ] **Chunking strategy** — your Strategy B/C design choices and why
- [ ] **Retrieval method** — BM25 vs TF-IDF choice, tokenization decisions,
  single vs. multiple indexes
- [ ] **Performance analysis** — your actual recall@1/3/5/10 numbers, split by
  docs vs. code, plus indexing time / cold-start / warm-throughput measurements
- [ ] **Design decisions** — the tradeoffs flagged throughout this plan (file
  filtering, chunk model shape, overlap counting convention, etc.)
- [ ] **Challenges faced** — real difficulties + how you resolved them
- [ ] **Example usage** — copy-pasteable commands, ideally mirroring the §V.6
  transcripts in the subject
- [ ] Written entirely in English

---

## 11. Pre-submission verification checklist

Run through this explicitly before considering the mandatory part done:

- [ ] `index` completes in under 5 minutes on the full vLLM repo
- [ ] Cold-start latency (first retrieval after startup, including model load)
  under 60 seconds
- [ ] Warm throughput: 1000 questions processed in under 90 seconds after cold
  start
- [ ] Recall@5 ≥ 80% on docs questions, ≥ 50% on code questions (measured with
  your own `evaluate`, against the public `AnsweredQuestions` datasets)
- [ ] `flake8 .` clean
- [ ] `mypy .` clean with the mandatory flags (and ideally `--strict`)
- [ ] CLI doesn't crash on: empty query, `k=0`, negative `k`, nonexistent file
  paths, malformed JSON input, an empty dataset
- [ ] Output JSON validates against `StudentSearchResults` /
  `StudentSearchResultsAndAnswer` exactly (field names, types, nesting)
- [ ] `data/` (or wherever generated artifacts live) is gitignored — don't commit
  the repo zip, the index, or generated outputs
- [ ] You can explain, out loud, every design decision in your README without
  looking at your own code — this is what the recode/defense step (§IX.1) is
  actually testing
