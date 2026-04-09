import os
import regex as re
import time
import pickle
from typing import BinaryIO
from collections import defaultdict
from collections.abc import Iterable, Iterator
from multiprocessing import Pool



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
    
    GPT2_pattern = [r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""]
    
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

def update_pair_counts(byte_pair_frequencies, chunk, freq, delta):
    for pair in zip(chunk, chunk[1:]):
        byte_pair_frequencies[pair] += delta * freq
        if byte_pair_frequencies[pair] == 0:
            del byte_pair_frequencies[pair]
    return byte_pair_frequencies

def merge(regex_chunk_table, byte_pair_frequencies, merges):
    pair_to_merge, new_token = merges[-1]
    new_regex_chunk_table = {}

    for chunk, freq in regex_chunk_table.items():
        new_chunk, changed = merge_chunk(chunk, pair_to_merge, new_token)

        if changed:
            byte_pair_frequencies = update_pair_counts(byte_pair_frequencies, chunk, freq, -1)
            byte_pair_frequencies = update_pair_counts(byte_pair_frequencies, new_chunk, freq, 1)

        new_regex_chunk_table[new_chunk] = new_regex_chunk_table.get(new_chunk, 0) + freq

    return new_regex_chunk_table, byte_pair_frequencies

def resolve_token(token, vocab):
    value = vocab[token]
    if isinstance(value, bytes):
        return value
    left, right = value
    return resolve_token(left, vocab) + resolve_token(right, vocab)

def pretokenization_work(input_path, start, end, special_tokens):
    local_regex_chunk_table = defaultdict(int)
    local_byte_pair_frequencies = defaultdict(int)

    with open(input_path, "rb") as f:   
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    for regex_chunk in pretokenize(chunk, special_tokens):
        regex_chunk_bytes = tuple(regex_chunk.encode("utf-8"))
        local_regex_chunk_table[regex_chunk_bytes] += 1
        for i in zip(regex_chunk_bytes, regex_chunk_bytes[1:]):
            local_byte_pair_frequencies[i] += 1

    return local_regex_chunk_table, local_byte_pair_frequencies

def BPE_Tokenizer_Training(input_path: str, vocab_size: int, special_tokens: list[str], parallelize: bool = False):
    """BPE Tokenizer training that allows for parallelizable training
    Args: 
        input_path (str): path to a txt file
        vocab size (int): desired final vocab_size, must be more than 256 as that is default for utf-8 bytes
        special_tokens (list[str]): list of special tokens that will not be split in final vocab
        parralelize (bool): whether to parrallelize the pretokenization work on all but one core of your device
    Returns:  
        vocab (dict[int, bytes]): the integer id to bytes of our final vocabulary
        merges (list[tuple[bytes,bytes]]): a chronological (lower index = earlier) list of merges of pairs of bytes
    """

    with open(input_path, "rb") as f:
        num_processes = max(1, (os.process_cpu_count() or 1) - 1)
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    
    print(f'Time at start: {time.perf_counter()}')
    
    vocab = vocab_init(special_tokens)
    regex_chunk_table: dict[tuple[bytes], int] = defaultdict(int) # this will store all regex chunks and their frequencies 
    byte_pair_frequencies: dict[tuple[bytes], int] = defaultdict(int)
    merges: list[(tuple[bytes], bytes)] = []

    print(f'Time before pretok: {time.perf_counter()}')

    if parallelize:
        tasks = [
            (input_path, start, end, special_tokens)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
        with Pool(num_processes) as p:
                results = p.starmap(pretokenization_work, tasks)    
        # combine worker outputs
        for local_regex_chunk_table, local_byte_pair_frequencies in results:
            for k, v in local_regex_chunk_table.items():
                regex_chunk_table[k] += v
            for k, v in local_byte_pair_frequencies.items():
                byte_pair_frequencies[k] += v
    else: 
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            local_regex_chunk_table, local_byte_pair_frequencies = pretokenization_work(input_path, start, end, special_tokens)

            for k, v in local_regex_chunk_table.items():
                regex_chunk_table[k] += v
            for k, v in local_byte_pair_frequencies.items():
                byte_pair_frequencies[k] += v
    
    print(f'Time after pretok: {time.perf_counter()}')


    while len(vocab) < vocab_size: 

        best_pair, best_pair_frequency = max(byte_pair_frequencies.items(), key=lambda x: (x[1], (vocab[x[0][0]], vocab[x[0][1]]))) #max function will lexographically tiebreak
        new_token_id = max(vocab.items(), key=lambda x: x[0])[0]+1
            
        left, right = best_pair
        vocab[new_token_id] = resolve_token(left, vocab) + resolve_token(right, vocab)
        merges.append((best_pair, new_token_id))
        regex_chunk_table, byte_pair_frequencies = merge(regex_chunk_table, byte_pair_frequencies, merges)

    print(f'Time after merges: {time.perf_counter()}')

    merges_without_id: list[tuple[bytes, bytes]] = [(vocab[m[0][0]], vocab[m[0][1]])  for m in merges]
    return vocab, merges_without_id
                


class Tokenizer():
    """Tokenizer class that has encoding, decoding"""

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = [], parallelize: bool = False):
        self.vocab = vocab
        self.reverse_vocab: dict[bytes, int] = {v:k for k, v in self.vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens
        if special_tokens is None:
            self.special_tokens = []
        self.special_tokens.sort(key = lambda x: len(x), reverse=True) #sorts by length for overlapping token edge case is handeled in encode_iterable

        self.tuple_special_tokens: set[tuple[bytes, ...]] = {
            tuple(bytes([b]) for b in special_token.encode("utf-8"))
            for special_token in self.special_tokens}
        self.parallelize = parallelize
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None, parallelize: bool = False):
        """method that constructs and return a Tokenizer from a serialized vocabulary and list of merge"""
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return Tokenizer(vocab, merges, special_tokens, parallelize)
    
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
        # TODO TODO index my adjacencies so i don't have to loop over corpus each time bit can go directly to places counts change
        merges_remaining = self.merges.copy()
        corpus: list[tuple[bytes, ...]] = []
        out: list[int] = []

        for regex_chunk in pretokenize(text, self.special_tokens, encoding=True):
            corpus.append(tuple(bytes([b]) for b in regex_chunk.encode('utf-8')))

        while len(merges_remaining) > 0: 
            for i, mini_text in enumerate(corpus): 
                if mini_text in self.tuple_special_tokens:
                    continue
                else: 
                    corpus[i] = self.replace_pair(mini_text, merges_remaining[0], merges_remaining[0][0]+merges_remaining[0][1])
            del merges_remaining[0]

        for i, mini_text in enumerate(corpus): 
            if mini_text in self.tuple_special_tokens:
                out.append(self.reverse_vocab[b''.join(mini_text)])
            else:
                for j in mini_text:
                    out.append(self.reverse_vocab[j])
        
        return out

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        
        Not sure if this is what it wants, because you would ***NEED TO*** break up the text at places where you know no
        merges will cross (ex: special tokens) in your iterable input
        
        But also this breaks it up into even smaller sections now
        """
        #TODO parallelize
        buffer = ""

        for chunk in iterable:
            buffer += chunk

            while any(tok in buffer for tok in self.special_tokens):
                indexes = sorted([(buffer.index(tok), tok) for tok in self.special_tokens if tok in buffer], key=lambda x: x[0])
                idx = indexes[0][0]+len(indexes[0][1])
                safe_text = buffer[:idx]
                buffer = buffer[idx:]
                yield from self.encode(safe_text)

        if buffer:
            yield from self.encode(buffer)

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        #TODO: parallelize + make a decode_iterable
        out = b""
        for i in ids: 
            out += self.vocab[i]
        return out.decode("utf-8", errors='replace')
    

#v, mwi = BPE_Tokenizer_Training("data/TinyStoriesV2-GPT4-valid.txt", 300, ["<|endoftext|>"])
