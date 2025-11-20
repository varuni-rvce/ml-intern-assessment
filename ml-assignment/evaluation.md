# Evaluation — Trigram Model (1‑page summary)

Design goals
- Implement a clear, minimal trigram language model that trains on raw text, handles basic cleaning and sentence boundaries, and generates probabilistic text.

Data structures
- counts: defaultdict(lambda: defaultdict(int)) mapping context (w1, w2) -> {next_word: count}. This is memory efficient for sparse n‑gram data and fast to update.
- vocab: set of observed tokens used for bookkeeping and potential future OOV handling.
- trained: boolean flag to indicate model readiness.

Preprocessing
- Tokenization: simple rule-based tokenizer:
  - Lowercase text.
  - Replace non-alphanumeric characters with spaces (regex: [^a-z0-9\s]).
  - Split on whitespace.
- Sentence segmentation: split on punctuation characters .!? to create sentence-level contexts.

Padding and sentence boundaries
- Use explicit start and end tokens: two start tokens '<s>', '<s>' then tokens then one end token '</s>'.
- Padding with two start tokens ensures the first real token is produced as a trigram continuation.

Training (fit)
- For each sentence, produce padded token list then iterate trigrams:
  - context = (token[i], token[i+1])
  - counts[context][token[i+2]] += 1
- Update vocab with observed tokens (excluding '</s>').

Generation (generate)
- Start from context ('<s>','<s>') and repeatedly:
  - Retrieve choices for current context; break if none.
  - Convert counts to sampling weights and choose next_word with Python's random.choices(weights=counts).
  - Stop if next_word == '</s>' or max_length reached.
- This produces probabilistic sampling proportional to observed counts (empirical MLE without smoothing).

Limitations and trade-offs
- No smoothing / backoff: unseen contexts lead to early termination. This keeps the implementation simple but reduces coverage.
- OOV handling: currently there is no explicit unknown token — unseen words simply won't be generated; training ignores empty input.
- Tokenization is simplistic and may conflate words with punctuation (adequate for assignment scope).
- Generation is non-deterministic unless a global random seed is set externally.

Potential improvements
- Add add‑one (Laplace) smoothing or Katz/Good‑Turing and backoff to handle unseen contexts.
- Introduce an explicit <UNK> token and a preprocessing pass to map infrequent words to UNK.
- Add a deterministic seed parameter to generate for reproducible outputs.
- Persist/load model to disk (JSON/pickle) and generalize to N-grams (N parameter).

How to test / reproduce
1. Install dependencies: python -m pip install -r requirements.txt
2. Run tests from package root:
   - cd ml-assignment
   - python -m pytest -q
3. Example quick test in Python REPL shown in README.md.

Conclusion
- The implementation prioritizes clarity and correctness for trigram counting and probabilistic generation. It is a concise base for incremental enhancements such as smoothing, OOV handling and model persistence.