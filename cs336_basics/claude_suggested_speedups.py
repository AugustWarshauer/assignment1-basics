"""Speedups for tokenizer.py, written by Claude. tokenizer.py imports these."""

import heapq
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import regex as re

# GPT-2's contraction alternative '(?:[sdmt]|ll|ve|re) reads up to 3 chars from where it starts. Cut a stream closer
# than that to a pretoken's start and "'ve" splits into "'" + "ve".
PRETOKEN_REACH = 4


class PretokenEncoder:
    """BPE-encodes one pretoken at a time, caching results by pretoken string.

    Repeatedly applying the lowest-ranked merge present gives the same result as replaying the whole merge list in
    order: a merge can't create a pair ranked below itself, since every new pair contains the token it just made.
    """

    def __init__(self, merges: list[tuple[bytes, bytes]], reverse_vocab: dict[bytes, int], replace_pair: Callable, cache_size: int):
        self.ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.reverse_vocab = reverse_vocab
        self.replace_pair = replace_pair
        self.cache: dict[str, list[int]] = {}
        self.cache_size = cache_size

    def __call__(self, pretoken: str) -> list[int]:
        cached = self.cache.get(pretoken)
        if cached is not None:
            return cached

        symbols = tuple(bytes([b]) for b in pretoken.encode("utf-8"))
        while len(symbols) > 1:
            best_rank, best_pair = None, None
            for pair in zip(symbols, symbols[1:]):
                rank = self.ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pair = rank, pair
            if best_pair is None:
                break
            symbols = self.replace_pair(symbols, best_pair, best_pair[0] + best_pair[1])

        ids = [self.reverse_vocab[s] for s in symbols]
        if len(self.cache) >= self.cache_size:
            self.cache.clear()
        self.cache[pretoken] = ids
        return ids


def stream_encode(
    iterable: Iterable[str],
    encode: Callable[[str], list[int]],
    special_tokens: list[str],
    pretoken_pattern: str,
    read_chars: int = 16_384,
    max_buffer_chars: int = 1 << 20,
) -> Iterator[int]:
    """Lazily encode a stream of text, handing `encode` only pieces that end at safe cut points.

    A cut is safe after a special token far enough from the end of the buffer that more text can't extend it into a
    longer overlapping one, or, on a long stretch with no special token, at a pretoken boundary at least
    PRETOKEN_REACH characters short of the end.
    """
    special_re = re.compile("|".join(re.escape(tok) for tok in special_tokens)) if special_tokens else None
    pretoken_re = re.compile(pretoken_pattern)
    max_special_len = max((len(tok) for tok in special_tokens), default=0)
    guard = 2 * max_special_len

    buffer = ""
    scan_from = 0
    for text in _batched(iterable, read_chars):
        buffer += text

        if special_re is not None:
            cut = 0
            for m in special_re.finditer(buffer, scan_from):
                if m.end() <= len(buffer) - max_special_len:
                    cut = m.end()
            if cut:
                yield from encode(buffer[:cut])
                buffer = buffer[cut:]
            scan_from = max(0, len(buffer) - guard)

        # limit must stay positive: the regex module reads a negative endpos as counting back from the end
        limit = len(buffer) - guard
        if len(buffer) >= max_buffer_chars and limit > 0:
            cut = 0
            for m in pretoken_re.finditer(buffer, 0, limit):
                if m.start() > limit - PRETOKEN_REACH:
                    break
                cut = m.start()
            if cut > 0:
                yield from encode(buffer[:cut])
                buffer = buffer[cut:]
                scan_from = max(0, len(buffer) - guard)

    if buffer:
        yield from encode(buffer)


def _batched(iterable: Iterable[str], min_chars: int) -> Iterator[str]:
    pending: list[str] = []
    n = 0
    for piece in iterable:
        pending.append(piece)
        n += len(piece)
        if n >= min_chars:
            yield "".join(pending)
            pending, n = [], 0
    if pending:
        yield "".join(pending)


class _Descending:
    """Inverts comparison so heapq, a min-heap, pops the lexicographically greatest bytes first on a count tie."""

    __slots__ = ("key",)

    def __init__(self, key):
        self.key = key

    def __lt__(self, other):
        return self.key > other.key

    def __eq__(self, other):
        return self.key == other.key


def run_bpe_merges(regex_chunk_table: dict[tuple[int, ...], int], vocab: dict[int, bytes], vocab_size: int, merge_chunk: Callable) -> list:
    """Merge until vocab reaches vocab_size, adding each new token to `vocab` in place.

    Returns merges as [((left_id, right_id), new_id), ...]. An index from each pair to the pretokens containing it means
    a merge only visits those pretokens, and a heap with lazy deletion finds the most frequent pair.
    """
    words = list(regex_chunk_table.keys())
    freqs = list(regex_chunk_table.values())
    pair_counts, pair_to_words = _build_pair_index(words, freqs)

    def heap_entry(pair):
        return (-pair_counts[pair], _Descending((vocab[pair[0]], vocab[pair[1]])), pair)

    heap = [heap_entry(pair) for pair in pair_counts]
    heapq.heapify(heap)
    merges = []

    while len(vocab) < vocab_size:
        best_pair = None
        while heap:
            neg_count, _, pair = heapq.heappop(heap)
            if pair_counts.get(pair) == -neg_count:
                best_pair = pair
                break
        if best_pair is None:
            break

        new_token_id = len(vocab)
        left, right = best_pair
        vocab[new_token_id] = vocab[left] + vocab[right]
        merges.append((best_pair, new_token_id))

        changed = _merge_pair_everywhere(best_pair, new_token_id, words, freqs, pair_counts, pair_to_words, merge_chunk)
        for pair, d in changed.items():
            # push decreases too, or a pair whose count dropped is left with only stale entries and never picked again
            if d != 0 and pair in pair_counts:
                heapq.heappush(heap, heap_entry(pair))

    return merges


def _build_pair_index(words, freqs):
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    pair_to_words: dict[tuple[int, int], set[int]] = defaultdict(set)
    for wi, (word, freq) in enumerate(zip(words, freqs)):
        for pair in zip(word, word[1:]):
            pair_counts[pair] += freq
            pair_to_words[pair].add(wi)
    return pair_counts, pair_to_words


def _merge_pair_everywhere(pair, new_token, words, freqs, pair_counts, pair_to_words, merge_chunk):
    delta: dict[tuple[int, int], int] = defaultdict(int)
    for wi in pair_to_words.pop(pair, ()):
        old_word, freq = words[wi], freqs[wi]
        new_word, _ = merge_chunk(old_word, pair, new_token)

        old_pairs = list(zip(old_word, old_word[1:]))
        new_pairs = list(zip(new_word, new_word[1:]))
        for p in old_pairs:
            delta[p] -= freq
        for p in new_pairs:
            delta[p] += freq

        for p in set(old_pairs).difference(new_pairs):
            if p != pair:
                pair_to_words[p].discard(wi)
        for p in new_pairs:
            pair_to_words[p].add(wi)

        words[wi] = new_word

    for p, d in delta.items():
        count = pair_counts.get(p, 0) + d
        if count > 0:
            pair_counts[p] = count
        else:
            pair_counts.pop(p, None)
            pair_to_words.pop(p, None)
    return delta


# ---------------------------------------------------------------- parallel corpus tokenization (used by modal_app.py)

_TOKENIZER = None


def _init_tokenizer(tokenizer_dir: str):
    global _TOKENIZER
    from cs336_basics.tokenizer import Tokenizer

    d = Path(tokenizer_dir)
    _TOKENIZER = Tokenizer.from_files(d / "vocab.pkl", d / "merges.pkl", ["<|endoftext|>"], pretoken_cache_size=200_000)


def _encode_range(task):
    import numpy as np

    txt, start, end, shard = task
    with open(txt, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    ids = np.array(_TOKENIZER.encode(text), dtype=np.int64)
    if ids.size and ids.max() >= 1 << 16:
        raise ValueError(f"token id {ids.max()} does not fit in uint16")
    ids.astype(np.uint16).tofile(shard)
    return shard, int(ids.size)


def tokenize_file(txt: Path, tokenizer_dir: Path, out: Path, workers: int, chunk_bytes: int = 16 << 20) -> int:
    """Tokenize `txt` into a flat uint16 file, in parallel.

    Chunks start at <|endoftext|>, and pretokenization splits on special tokens first, so encoding the chunks separately
    gives exactly the ids that encoding the whole file would.
    """
    from multiprocessing import Pool

    from cs336_basics.tokenizer import find_chunk_boundaries

    with open(txt, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        bounds = find_chunk_boundaries(f, max(1, size // chunk_bytes), b"<|endoftext|>")
    shard_dir = Path(tempfile.mkdtemp())
    tasks = [(str(txt), s, e, str(shard_dir / f"{i:06d}.bin")) for i, (s, e) in enumerate(zip(bounds[:-1], bounds[1:]))]

    total = 0
    with Pool(workers, initializer=_init_tokenizer, initargs=(str(tokenizer_dir),)) as pool, open(out, "wb") as dst:
        for shard, n in pool.imap(_encode_range, tasks):
            with open(shard, "rb") as src:
                shutil.copyfileobj(src, dst)
            os.remove(shard)
            total += n
    shutil.rmtree(shard_dir)
    return total


def download_text(url: str, dest_dir: Path) -> Path:
    import gzip
    import urllib.request

    dest_dir.mkdir(parents=True, exist_ok=True)
    raw = dest_dir / url.rsplit("/", 1)[1]
    urllib.request.urlretrieve(url, raw)
    if raw.suffix != ".gz":
        return raw
    txt = raw.with_suffix("")
    with gzip.open(raw, "rb") as src, open(txt, "wb") as dst:
        shutil.copyfileobj(src, dst, 16 << 20)
    raw.unlink()
    return txt
