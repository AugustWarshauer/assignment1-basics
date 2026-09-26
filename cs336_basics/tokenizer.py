import os
import regex as re
import time
import pickle
from typing import BinaryIO
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from multiprocessing import Pool
from cs336_basics.claude_suggested_speedups import PretokenEncoder, run_bpe_merges, stream_encode

# module level so encode_iterable cuts the stream on exactly the same pretoken boundaries as pretokenize
GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pretokenize(chunk: str, special_tokens: list[str], encoding: bool = False) -> iter:
    

    GPToseries_pattern = [
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+"""
    ]
    
    GPT2_pattern = [GPT2_PAT]
    
    special_pat = "|".join(re.escape(tok) for tok in special_tokens)

    if len(special_tokens) == 0:
        parts = [chunk]
    elif not encoding: 
        parts = re.split(special_pat, chunk)
    else: 
        parts = re.split(f"({special_pat})", chunk)
    
    PAT = "|".join(GPT2_pattern)

    def _iter_tokens():
        for part in parts:
            if not part:
                continue

            if encoding and part in special_tokens:
                yield part
            else:
                for match in re.finditer(PAT, part):
                    yield match.group()

    return _iter_tokens()

def vocab_init(special_tokens: list[str]) -> dict[int, bytes]:
    vocab = {i: bytes([i]) for i in range(256)}

    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")

    return vocab

def merge_chunk(chunk, pair_to_merge, new_token):
    out = []
    i = 0
    changed = False

    while i < len(chunk):
        if i + 1 < len(chunk) and (chunk[i], chunk[i + 1]) == pair_to_merge:
            out.append(new_token)
            i += 2
            changed = True
        else:
            out.append(chunk[i])
            i += 1

    return tuple(out), changed

def pretokenization_work(task) -> dict[tuple[int, ...], int]:
    input_path, start, end, special_tokens = task
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    counts = Counter(pretokenize(chunk, special_tokens))
    return {tuple(pretoken.encode("utf-8")): n for pretoken, n in counts.items()}

def BPE_Tokenizer_Training(input_path: str, vocab_size: int, special_tokens: list[str], parallelize: bool = False, chunk_bytes: int = 32 * 1024 * 1024):
    """BPE Tokenizer training that allows for parallelizable training
    Args: 
        input_path (str): path to a txt file
        vocab size (int): desired final vocab_size, must be more than 256 as that is default for utf-8 bytes
        special_tokens (list[str]): list of special tokens that will not be split in final vocab
        parralelize (bool): whether to parrallelize the pretokenization work on all but one core of your device
        chunk_bytes (int): target size of each pretokenization chunk, which bounds memory per worker
    Returns:  
        vocab (dict[int, bytes]): the integer id to bytes of our final vocabulary
        merges (list[tuple[bytes,bytes]]): a chronological (lower index = earlier) list of merges of pairs of bytes
    """

    with open(input_path, "rb") as f:
        num_processes = max(1, (os.process_cpu_count() or 1) - 1)
        file_size = os.fstat(f.fileno()).st_size
        num_chunks = max(num_processes, file_size // chunk_bytes)
        boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")

    print(f'Time at start: {time.perf_counter()}')

    vocab = vocab_init(special_tokens)
    regex_chunk_table: dict[tuple[int, ...], int] = defaultdict(int) # this will store all regex chunks and their frequencies

    print(f'Time before pretok: {time.perf_counter()}')

    tasks = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    if parallelize:
        with Pool(num_processes) as p:
            for local_regex_chunk_table in p.imap_unordered(pretokenization_work, tasks):
                for k, v in local_regex_chunk_table.items():
                    regex_chunk_table[k] += v
    else:
        for task in tasks:
            for k, v in pretokenization_work(task).items():
                regex_chunk_table[k] += v

    print(f'Time after pretok: {time.perf_counter()}')

    merges = run_bpe_merges(regex_chunk_table, vocab, vocab_size, merge_chunk)

    print(f'Time after merges: {time.perf_counter()}')

    merges_without_id: list[tuple[bytes, bytes]] = [(vocab[m[0][0]], vocab[m[0][1]])  for m in merges]
    return vocab, merges_without_id
                

class Tokenizer():
    """Tokenizer class that has encoding, decoding"""

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None, parallelize: bool = False, pretoken_cache_size: int = 4096):
        self.vocab = vocab
        self.reverse_vocab: dict[bytes, int] = {v:k for k, v in self.vocab.items()}
        self.merges = merges
        self.special_tokens: list[str] = sorted(special_tokens or [], key=len, reverse=True) #sorts by length for overlapping token edge case
        self.special_token_set: set[str] = set(self.special_tokens)
        self.parallelize = parallelize
        self._pretoken_encoder = PretokenEncoder(self.merges, self.reverse_vocab, self.replace_pair, pretoken_cache_size)
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None, parallelize: bool = False, pretoken_cache_size: int = 4096):
        """method that constructs and return a Tokenizer from a serialized vocabulary and list of merge"""
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return Tokenizer(vocab, merges, special_tokens, parallelize, pretoken_cache_size)
    
    def replace_pair(self, seq: tuple[bytes, ...], pair: tuple[bytes, bytes], new: bytes):
        out: list = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and (seq[i], seq[i+1]) == pair:
                out.append(new)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return tuple(out)

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        #TODO: make parallelizable
        out: list[int] = []
        for pretoken in pretokenize(text, self.special_tokens, encoding=True):
            if pretoken in self.special_token_set:
                out.append(self.reverse_vocab[pretoken.encode("utf-8")])
            else:
                out.extend(self._pretoken_encoder(pretoken))
        return out

    def encode_iterable(self, iterable: Iterable[str], read_chars: int = 16_384, max_buffer_chars: int = 1 << 20) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        """
        #TODO parallelize
        yield from stream_encode(iterable, self.encode, self.special_tokens, GPT2_PAT, read_chars, max_buffer_chars)

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        #TODO: parallelize + make a decode_iterable
        out = b""
        for i in ids: 
            out += self.vocab[i]
        return out.decode("utf-8", errors='replace')
    

#v, mwi = BPE_Tokenizer_Training("data/TinyStoriesV2-GPT4-valid.txt", 300, ["<|endoftext|>"])
