import builtins
import collections
from collections import OrderedDict, UserDict
import unicodedata
import io
import bisect
from enum import Enum
import json
import numpy as np
import os
import warnings
import re
from pprint import pprint
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
import math
from dataclasses import dataclass, field

# from sqlalchemy import false
from uie_acl import _data_interaction_in, _gen_dataset, inference, _destroy_data_set_buffer, _data_interaction_out, print_result, release, allocate_res, load_model, get_model_data, malloc_device

MODEL_PATH = "/root/lijiaqi/UIE/model/model_uie_v2.om"

class ExplicitEnum(Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


@dataclass
class FasterEncoding:
    """This is dummy class reserved for fast tokenizer"""

    pass

@dataclass(frozen=True, eq=True)
class AddedToken:
    """
    AddedToken represents a token to be added to a Tokenizer An AddedToken can have special options defining the
    way it should behave.
    """

    content: str = field(default_factory=str)
    single_word: bool = False
    lstrip: bool = False
    rstrip: bool = False
    normalized: bool = True

    def __getstate__(self):
        return self.__dict__

    def __str__(self):
        return self.content


def to_py_obj(obj):
    """
    Convert Numpy array or python list to a python list.
    """
    if isinstance(obj, (dict, UserDict)):
        return {k: to_py_obj(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_py_obj(o) for o in obj]
    elif isinstance(
            obj, (np.ndarray, np.number)):  # tolist also works on 0d np arrays
        return obj.tolist()
    else:
        return obj


def _is_numpy(x):
    return isinstance(x, np.ndarray)


class TruncationStrategy(ExplicitEnum):
    """
    Possible values for the `truncation` argument in [`PretrainedTokenizerBase.__call__`]. Useful for tab-completion in
    an IDE.
    """

    ONLY_FIRST = "only_first"
    ONLY_SECOND = "only_second"
    LONGEST_FIRST = "longest_first"
    DO_NOT_TRUNCATE = "do_not_truncate"


class CharSpan(NamedTuple):
    """
    Character span in the original string.

    Args:
        start (`int`): Index of the first character in the original string.
        end (`int`): Index of the character following the last character in the original string.
    """

    start: int
    end: int


class TokenSpan(NamedTuple):
    """
    Token span in an encoded string (list of tokens).

    Args:
        start (`int`): Index of the first token in the span.
        end (`int`): Index of the token following the last token in the span.
    """

    start: int
    end: int


class BatchEncoding(UserDict):
    """
    Holds the output of the [`PretrainedTokenizerBase.__call__`],
    [`PretrainedTokenizerBase.encode_plus`] and
    [`PretrainedTokenizerBase.batch_encode_plus`] methods (tokens, attention_masks, etc).

    This class is derived from a python dictionary and can be used as a dictionary. In addition, this class exposes
    utility methods to map from word/character space to token space.

    Args:
        data (`dict`):
            Dictionary of lists/arrays/tensors returned by the `__call__`/`encode`/`batch_encode` methods
            ('input_ids', 'attention_mask', etc.).
        tensor_type (`Union[None, str, TensorType]`, *optional*):
            You can give a tensor_type here to convert the lists of integers in Paddle/Numpy Tensors at
            initialization.
        prepend_batch_axis (`bool`, *optional*, defaults to `False`):
            Whether or not to add a batch axis when converting to tensors (see `tensor_type` above).
    """

    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None,
        encoding: Optional[Union[FasterEncoding,
                                 Sequence[FasterEncoding]]] = None,
        tensor_type: Union[None, str] = None,
        prepend_batch_axis: bool = False,
        n_sequences: Optional[int] = None,
    ):
        super().__init__(data)

        if isinstance(encoding, FasterEncoding):
            encoding = [encoding]

        self._encodings = encoding

        if n_sequences is None and encoding is not None and len(encoding):
            n_sequences = encoding[0].n_sequences

        self._n_sequences = n_sequences

        self.convert_to_tensors(tensor_type=tensor_type,
                                prepend_batch_axis=prepend_batch_axis)

    @property
    def n_sequences(self) -> Optional[int]:
        """
        `Optional[int]`: The number of sequences used to generate each sample from the batch encoded in this
        [`BatchEncoding`]. Currently can be one of `None` (unknown), `1` (a single sentence) or `2` (a pair of
        sentences)
        """
        return self._n_sequences

    @property
    def is_fast(self) -> bool:
        """
        `bool`: Indicate whether this [`BatchEncoding`] was generated from the result of a [`PretrainedFasterTokenizer`]
        or not.
        """
        return self._encodings is not None

    def __getitem__(self, item: Union[int, str]) -> Union[Any, FasterEncoding]:
        """
        If the key is a string, returns the value of the dict associated to `key` ('input_ids', 'attention_mask',
        etc.).

        If the key is an integer, get the `Encoding` for batch item with index `key`.
        """
        if isinstance(item, str):
            return self.data[item]
        elif self._encodings is not None:
            return self._encodings[item]
        else:
            raise KeyError(
                "Indexing with integers is not available when using tokenizer.__call__()"
                " with return_dict=True. Please set return_dict to False to use integer indexing."
            )

    def __getattr__(self, item: str):
        try:
            return self.data[item]
        except KeyError:
            raise AttributeError

    def __getstate__(self):
        return {"data": self.data, "encodings": self._encodings}

    def __setstate__(self, state):
        if "data" in state:
            self.data = state["data"]

        if "encodings" in state:
            self._encodings = state["encodings"]

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    # After this point:
    # Extended properties and methods only available for fast tokenizers
    # not yet supported

    @property
    def encodings(self) -> Optional[List[FasterEncoding]]:
        """
        `Optional[List[FasterEncoding]]`: The list all encodings from the tokenization process. Returns `None` if
        the input was tokenized through Python (i.e., not a fast) tokenizer.
        """
        return self._encodings

    def tokens(self, batch_index: int = 0) -> List[str]:
        """
        Return the list of tokens (sub-parts of the input strings after word/subword splitting and before conversion to
        integer indices) at a given batch index (only works for the output of a fast tokenizer).

        Args:
            batch_index (`int`, *optional*, defaults to 0): The index to access in the batch.

        Returns:
            `List[str]`: The list of tokens at that index.
        """
        if not self._encodings:
            raise ValueError(
                "tokens() is not available when using Python-based tokenizers")
        return self._encodings[batch_index].tokens

    def sequence_ids(self, batch_index: int = 0) -> List[Optional[int]]:
        """
        Return a list mapping the tokens to the id of their original sentences:

            - `None` for special tokens added around or between sequences,
            - `0` for tokens corresponding to words in the first sequence,
            - `1` for tokens corresponding to words in the second sequence when a pair of sequences was jointly
              encoded.

        Args:
            batch_index (`int`, *optional*, defaults to 0): The index to access in the batch.

        Returns:
            `List[Optional[int]]`: A list indicating the sequence id corresponding to each token. Special tokens added
            by the tokenizer are mapped to `None` and other tokens are mapped to the index of their corresponding
            sequence.
        """
        if not self._encodings:
            raise ValueError(
                "sequence_ids() is not available when using Python-based tokenizers"
            )
        return self._encodings[batch_index].sequence_ids

    def words(self, batch_index: int = 0) -> List[Optional[int]]:
        """
        Return a list mapping the tokens to their actual word in the initial sentence for a fast tokenizer.

        Args:
            batch_index (`int`, *optional*, defaults to 0): The index to access in the batch.

        Returns:
            `List[Optional[int]]`: A list indicating the word corresponding to each token. Special tokens added by the
            tokenizer are mapped to `None` and other tokens are mapped to the index of their corresponding word
            (several tokens will be mapped to the same word index if they are parts of that word).
        """
        if not self._encodings:
            raise ValueError(
                "words() is not available when using Python-based tokenizers")
        warnings.warn(
            "`BatchEncoding.words()` property is deprecated and should be replaced with the identical, "
            "but more self-explanatory `BatchEncoding.word_ids()` property.",
            FutureWarning,
        )
        return self.word_ids(batch_index)

    def word_ids(self, batch_index: int = 0) -> List[Optional[int]]:
        """
        Return a list mapping the tokens to their actual word in the initial sentence for a fast tokenizer.

        Args:
            batch_index (`int`, *optional*, defaults to 0): The index to access in the batch.

        Returns:
            `List[Optional[int]]`: A list indicating the word corresponding to each token. Special tokens added by the
            tokenizer are mapped to `None` and other tokens are mapped to the index of their corresponding word
            (several tokens will be mapped to the same word index if they are parts of that word).
        """
        if not self._encodings:
            raise ValueError(
                "word_ids() is not available when using Python-based tokenizers"
            )
        return self._encodings[batch_index].word_ids

    def token_to_sequence(self,
                          batch_or_token_index: int,
                          token_index: Optional[int] = None) -> int:
        """
        Get the index of the sequence represented by the given token. In the general use case, this method returns `0`
        for a single sequence or the first sequence of a pair, and `1` for the second sequence of a pair

        Can be called as:

        - `self.token_to_sequence(token_index)` if batch size is 1
        - `self.token_to_sequence(batch_index, token_index)` if batch size is greater than 1

        This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e.,
        words are defined by the user). In this case it allows to easily associate encoded tokens with provided
        tokenized words.

        Args:
            batch_or_token_index (`int`):
                Index of the sequence in the batch. If the batch only comprises one sequence, this can be the index of
                the token in the sequence.
            token_index (`int`, *optional*):
                If a batch index is provided in *batch_or_token_index*, this can be the index of the token in the
                sequence.

        Returns:
            `int`: Index of the word in the input sequence.
        """

        if not self._encodings:
            raise ValueError(
                "token_to_sequence() is not available when using Python based tokenizers"
            )
        if token_index is not None:
            batch_index = batch_or_token_index
        else:
            batch_index = 0
            token_index = batch_or_token_index
        if batch_index < 0:
            batch_index = self._batch_size + batch_index
        if token_index < 0:
            token_index = self._seq_len + token_index
        return self._encodings[batch_index].token_to_sequence(token_index)

    def token_to_word(self,
                      batch_or_token_index: int,
                      token_index: Optional[int] = None) -> int:
        """
        Get the index of the word corresponding (i.e. comprising) to an encoded token in a sequence of the batch.

        Can be called as:

        - `self.token_to_word(token_index)` if batch size is 1
        - `self.token_to_word(batch_index, token_index)` if batch size is greater than 1

        This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e.,
        words are defined by the user). In this case it allows to easily associate encoded tokens with provided
        tokenized words.

        Args:
            batch_or_token_index (`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
                the token in the sequence.
            token_index (`int`, *optional*):
                If a batch index is provided in *batch_or_token_index*, this can be the index of the token in the
                sequence.

        Returns:
            `int`: Index of the word in the input sequence.
        """

        if not self._encodings:
            raise ValueError(
                "token_to_word() is not available when using Python based tokenizers"
            )
        if token_index is not None:
            batch_index = batch_or_token_index
        else:
            batch_index = 0
            token_index = batch_or_token_index
        if batch_index < 0:
            batch_index = self._batch_size + batch_index
        if token_index < 0:
            token_index = self._seq_len + token_index
        return self._encodings[batch_index].token_to_word(token_index)

    def word_to_tokens(self,
                       batch_or_word_index: int,
                       word_index: Optional[int] = None,
                       sequence_index: int = 0) -> Optional[TokenSpan]:
        """
        Get the encoded token span corresponding to a word in a sequence of the batch.

        Token spans are returned as a [`TokenSpan`] with:

        - **start** -- Index of the first token.
        - **end** -- Index of the token following the last token.

        Can be called as:

        - `self.word_to_tokens(word_index, sequence_index: int = 0)` if batch size is 1
        - `self.word_to_tokens(batch_index, word_index, sequence_index: int = 0)` if batch size is greater or equal to
          1

        This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e. words
        are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized
        words.

        Args:
            batch_or_word_index (`int`):
                Index of the sequence in the batch. If the batch only comprises one sequence, this can be the index of
                the word in the sequence.
            word_index (`int`, *optional*):
                If a batch index is provided in *batch_or_token_index*, this can be the index of the word in the
                sequence.
            sequence_index (`int`, *optional*, defaults to 0):
                If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0
                or 1) the provided word index belongs to.

        Returns:
            Optional [`TokenSpan`] Span of tokens in the encoded sequence. Returns `None` if
            no tokens correspond to the word.
        """

        if not self._encodings:
            raise ValueError(
                "word_to_tokens() is not available when using Python based tokenizers"
            )
        if word_index is not None:
            batch_index = batch_or_word_index
        else:
            batch_index = 0
            word_index = batch_or_word_index
        if batch_index < 0:
            batch_index = self._batch_size + batch_index
        if word_index < 0:
            word_index = self._seq_len + word_index
        span = self._encodings[batch_index].word_to_tokens(
            word_index, sequence_index)
        return TokenSpan(*span) if span is not None else None

    def token_to_chars(self,
                       batch_or_token_index: int,
                       token_index: Optional[int] = None) -> CharSpan:
        """
        Get the character span corresponding to an encoded token in a sequence of the batch.

        Character spans are returned as a [`CharSpan`] with:

        - **start** -- Index of the first character in the original string associated to the token.
        - **end** -- Index of the character following the last character in the original string associated to the
          token.

        Can be called as:

        - `self.token_to_chars(token_index)` if batch size is 1
        - `self.token_to_chars(batch_index, token_index)` if batch size is greater or equal to 1

        Args:
            batch_or_token_index (`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
                the token in the sequence.
            token_index (`int`, *optional*):
                If a batch index is provided in *batch_or_token_index*, this can be the index of the token or tokens in
                the sequence.

        Returns:
            [`CharSpan`]: Span of characters in the original string.
        """

        if not self._encodings:
            raise ValueError(
                "token_to_chars() is not available when using Python based tokenizers"
            )
        if token_index is not None:
            batch_index = batch_or_token_index
        else:
            batch_index = 0
            token_index = batch_or_token_index
        return CharSpan(
            *(self._encodings[batch_index].token_to_chars(token_index)))

    def char_to_token(self,
                      batch_or_char_index: int,
                      char_index: Optional[int] = None,
                      sequence_index: int = 0) -> int:
        """
        Get the index of the token in the encoded output comprising a character in the original string for a sequence
        of the batch.

        Can be called as:

        - `self.char_to_token(char_index)` if batch size is 1
        - `self.char_to_token(batch_index, char_index)` if batch size is greater or equal to 1

        This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e. words
        are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized
        words.

        Args:
            batch_or_char_index (`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
                the word in the sequence
            char_index (`int`, *optional*):
                If a batch index is provided in *batch_or_token_index*, this can be the index of the word in the
                sequence.
            sequence_index (`int`, *optional*, defaults to 0):
                If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0
                or 1) the provided character index belongs to.


        Returns:
            `int`: Index of the token.
        """

        if not self._encodings:
            raise ValueError(
                "char_to_token() is not available when using Python based tokenizers"
            )
        if char_index is not None:
            batch_index = batch_or_char_index
        else:
            batch_index = 0
            char_index = batch_or_char_index
        return self._encodings[batch_index].char_to_token(
            char_index, sequence_index)

    def word_to_chars(self,
                      batch_or_word_index: int,
                      word_index: Optional[int] = None,
                      sequence_index: int = 0):
        """
        Get the character span in the original string corresponding to given word in a sequence of the batch.

        Character spans are returned as a CharSpan NamedTuple with:

        - start: index of the first character in the original string
        - end: index of the character following the last character in the original string

        Can be called as:

        - `self.word_to_chars(word_index)` if batch size is 1
        - `self.word_to_chars(batch_index, word_index)` if batch size is greater or equal to 1

        Args:
            batch_or_word_index (`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
                the word in the sequence
            word_index (`int`, *optional*):
                If a batch index is provided in *batch_or_token_index*, this can be the index of the word in the
                sequence.
            sequence_index (`int`, *optional*, defaults to 0):
                If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0
                or 1) the provided word index belongs to.

        Returns:
            `CharSpan` or `List[CharSpan]`: Span(s) of the associated character or characters in the string. CharSpan
            are NamedTuple with:

                - start: index of the first character associated to the token in the original string
                - end: index of the character following the last character associated to the token in the original
                  string
        """

        if not self._encodings:
            raise ValueError(
                "word_to_chars() is not available when using Python based tokenizers"
            )
        if word_index is not None:
            batch_index = batch_or_word_index
        else:
            batch_index = 0
            word_index = batch_or_word_index
        return CharSpan(*(self._encodings[batch_index].word_to_chars(
            word_index, sequence_index)))

    def char_to_word(self,
                     batch_or_char_index: int,
                     char_index: Optional[int] = None,
                     sequence_index: int = 0) -> int:
        """
        Get the word in the original string corresponding to a character in the original string of a sequence of the
        batch.

        Can be called as:

        - `self.char_to_word(char_index)` if batch size is 1
        - `self.char_to_word(batch_index, char_index)` if batch size is greater than 1

        This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e. words
        are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized
        words.

        Args:
            batch_or_char_index (`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
                the character in the original string.
            char_index (`int`, *optional*):
                If a batch index is provided in *batch_or_token_index*, this can be the index of the character in the
                original string.
            sequence_index (`int`, *optional*, defaults to 0):
                If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0
                or 1) the provided character index belongs to.


        Returns:
            `int` or `List[int]`: Index or indices of the associated encoded token(s).
        """

        if not self._encodings:
            raise ValueError(
                "char_to_word() is not available when using Python based tokenizers"
            )
        if char_index is not None:
            batch_index = batch_or_char_index
        else:
            batch_index = 0
            char_index = batch_or_char_index
        return self._encodings[batch_index].char_to_word(
            char_index, sequence_index)

    def convert_to_tensors(self, tensor_type: Optional[str] = None, prepend_batch_axis: bool = False):
        """
        Convert the inner content to tensors.

        Args:
            tensor_type (`str` or [`TensorType`], *optional*):
                The type of tensors to use. If `str`, should be one of the values of the enum [`TensorType`]. If
                `None`, no modification is done.
            prepend_batch_axis (`int`, *optional*, defaults to `False`):
                Whether or not to add the batch dimension during the conversion.
        """
        if tensor_type is None:
            return self

        # Convert to TensorType
        if not isinstance(tensor_type, TensorType):
            tensor_type = TensorType(tensor_type)
        # Get a function reference for the correct framework

        as_tensor = np.asarray
        is_tensor = _is_numpy

        # Do the tensor conversion in batch
        for key, value in self.items():
            try:
                if prepend_batch_axis:
                    value = [value]

                if not is_tensor(value):
                    tensor = as_tensor(value)

                    self[key] = tensor
            except:  # noqa E722
                if key == "overflowing_tokens":
                    raise ValueError(
                        "Unable to create tensor returning overflowing tokens of different lengths. "
                        "Please see if a fast version of this tokenizer is available to have this feature available."
                    )
                raise ValueError(
                    "Unable to create tensor, you should probably activate truncation and/or padding "
                    "with 'padding=True' 'truncation=True' to have batched tensors with the same length."
                )

        return self


class Trie:
    """
    Trie in Python. Creates a Trie out of a list of words. The trie is used to split on `added_tokens` in one pass
    Loose reference https://en.wikipedia.org/wiki/Trie
    """

    def __init__(self):
        self.data = {}

    def add(self, word: str):
        if not word:
            # Prevent empty string
            return
        ref = self.data
        for char in word:
            ref[char] = char in ref and ref[char] or {}
            ref = ref[char]
        ref[""] = 1

    def split(self, text: str) -> List[str]:
        """
        Will look for the words added to the trie within `text`. Output is the original string splitted along the
        boundaries of the words found.

        This trie will match the longest possible word first !
        """
       
        states = OrderedDict()
        offsets = [0]
        skip = 0
        # Main loop, Giving this algorithm O(n) complexity
        for current, current_char in enumerate(text):
            if skip and current < skip:
                # Prevents the lookahead for matching twice
                # like extra_id_100 and id_100
                continue
            to_remove = set()
            # Whenever we found a match, we need to drop everything
            # this is a greedy algorithm, it will match on the first found token
            reset = False

            # In this case, we already have partial matches (But unfinished)
            for start, trie_pointer in states.items():
                if "" in trie_pointer:                   
                    # "[CLS]", "L", we need to match CLS even if L is special
                    for lookstart, looktrie_pointer in states.items():
                        if lookstart > start:
                            # This partial match is later, we can stop looking
                            break
                        elif lookstart < start:
                            # This partial match is earlier, the trie pointer
                            # was already updated, so index is + 1
                            lookahead_index = current + 1
                            end = current + 1
                        else:
                            # Here lookstart == start and
                            #      looktrie_pointer == trie_pointer
                            # It wasn't updated yet so indices are current ones
                            lookahead_index = current
                            end = current
                        next_char = text[
                            lookahead_index] if lookahead_index < len(
                                text) else None
                        if "" in looktrie_pointer:
                            start = lookstart
                            end = lookahead_index
                            skip = lookahead_index

                        while next_char in looktrie_pointer:
                            looktrie_pointer = looktrie_pointer[next_char]
                            lookahead_index += 1
                            if "" in looktrie_pointer:
                                start = lookstart
                                end = lookahead_index
                                skip = lookahead_index

                            if lookahead_index == len(text):
                                # End of string
                                break
                            next_char = text[lookahead_index]
                        # End lookahead

                        # Storing and resetting
                    offsets.append(start)
                    offsets.append(end)
                    reset = True
                    break
                elif current_char in trie_pointer:
                    # The current character being looked at has a match within the trie
                    # update the pointer (it will be stored back into states later).
                    trie_pointer = trie_pointer[current_char]

                    # Storing back the new pointer into the states.
                    # Partial matches got longer by one.
                    states[start] = trie_pointer
                else:
                    to_remove.add(start)

            # Either clearing the full start (we found a real match)
            # Or clearing only the partial matches that didn't work.
            if reset:
                states = {}
            else:
                for start in to_remove:
                    del states[start]

            # If this character is a starting character within the trie
            # start keeping track of this partial match.
            if current >= skip and current_char in self.data:
                states[current] = self.data[current_char]

        # We have a cut at the end with states.
        for start, trie_pointer in states.items():
            if "" in trie_pointer:
                # This is a final match, we need to reset and
                # store the results in `offsets`.
                end = len(text)
                offsets.append(start)
                offsets.append(end)
                # Longest cut is always the one with lower start so the first
                # item so we need to break.
                break

        return self.cut_text(text, offsets)

    def cut_text(self, text, offsets):
        # We have all the offsets now, we just need to do the actual splitting.
        # We need to eventually add the first part of the string and the eventual
        # last part.
        offsets.append(len(text))
        tokens = []
        start = 0
        for end in offsets:
            if start > end:
                continue
            elif start == end:
                # This might happen if there's a match at index 0
                # we're also preventing zero-width cuts in case of two
                # consecutive matches
                continue
            tokens.append(text[start:end])
            start = end

        return tokens


def _insert_one_token_to_ordered_list(token_list: List[str], new_token: str):
    """
    Inserts one token to an ordered list if it does not already exist. Note: token_list must be sorted.
    """
    insertion_idx = bisect.bisect_left(token_list, new_token)
    # Checks if new_token is already in the ordered token_list
    if insertion_idx < len(
            token_list) and token_list[insertion_idx] == new_token:
        # new_token is in token_list, don't add
        return
    else:
        token_list.insert(insertion_idx, new_token)

def convert_to_unicode(text):
        """
        Converts `text` to Unicode (if it's not already), assuming utf-8 input.
        Args:
            text (str|bytes): Text to be converted to unicode.
        Returns:
            str: converted text.
        """
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))

def whitespace_tokenize(text):
    """
    Runs basic whitespace cleaning and splitting on a peice of text.
    Args:
        text (str): Text to be tokened.
    Returns:
        list(str): Token list.
    """
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

def _is_whitespace(char):
    """
    Checks whether `chars` is a whitespace character.
    """
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False

def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64)
            or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False

VERY_LARGE_INTEGER = int(
    1e30
)  # This is used to set the max input length for a model with infinite size input
LARGE_INTEGER = int(
    1e20
)  # This is used when we need something big but slightly smaller than VERY_LARGE_INTEGER

# Define type aliases and NamedTuples
TextInput = str
PreTokenizedInput = List[str]
EncodedInput = List[int]
TextInputPair = Tuple[str, str]
PreTokenizedInputPair = Tuple[List[str], List[str]]
EncodedInputPair = Tuple[List[int], List[int]]


class ExplicitEnum(Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class PaddingStrategy(ExplicitEnum):
    """
    Possible values for the `padding` argument in [`PretrainedTokenizerBase.__call__`]. Useful for tab-completion in an
    IDE.
    """

    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


class TensorType(ExplicitEnum):
    """
    Possible values for the `return_tensors` argument in [`PretrainedTokenizerBase.__call__`]. Useful for
    tab-completion in an IDE.
    """

    PADDLE = "pd"
    NUMPY = "np"


class TruncationStrategy(ExplicitEnum):
    """
    Possible values for the `truncation` argument in [`PretrainedTokenizerBase.__call__`]. Useful for tab-completion in
    an IDE.
    """

    ONLY_FIRST = "only_first"
    ONLY_SECOND = "only_second"
    LONGEST_FIRST = "longest_first"
    DO_NOT_TRUNCATE = "do_not_truncate"


class Vocab(object):
    """
    The class used to convert between tokens and ids. It also includes some 
    store/load functions.
    """

    def __init__(self,
                 counter=None,
                 max_size=None,
                 min_freq=1,
                 token_to_idx=None,
                 unk_token=None,
                 pad_token=None,
                 bos_token=None,
                 eos_token=None,
                 **kwargs):
        # Handle special tokens
        combs = (('unk_token', unk_token), ('pad_token', pad_token),
                 ('bos_token', bos_token), ('eos_token', eos_token))
        for name, value in combs:
            kwargs[name] = value
        special_tokens = []
        special_iter = kwargs.keys()
        # sort alphabetically
        special_iter = sorted(special_iter)
        for special_token_name in special_iter:
            # Test if kwarg specifies a special token
            if not special_token_name.endswith('_token'):
                raise ValueError(
                    '{} is invalid. Only keyword arguments '
                    'that end in \'_token\' are supported '
                    'to declare special tokens.'.format(special_token_name))

            special_token = kwargs[special_token_name]
            if special_token is not None and special_token not in special_tokens:
                special_tokens.append(special_token)

        if counter is None:
            # use token_to_idx as dict to import pretrained vocabulary
            assert token_to_idx, (
                'token_to_idx should not be None when counter is None')
            for special_token in special_tokens:
                assert special_token in token_to_idx, '{} is not in token_to_idx'.format(
                    special_token)
            self._token_to_idx = token_to_idx
            self._idx_to_token = {
                idx: token
                for token, idx in token_to_idx.items()
            }
            if unk_token:
                unk_index = self._token_to_idx[unk_token]
                self._token_to_idx = collections.defaultdict(lambda: unk_index)
                self._token_to_idx.update(token_to_idx)
        else:
            self._idx_to_token = {
                idx: special_token
                for idx, special_token in enumerate(special_tokens)
            }
            self._token_to_idx = collections.defaultdict()
            self._token_to_idx.update(
                (token, idx) for idx, token in self._idx_to_token.items())
            self._index_counter_keys(counter, special_tokens, max_size,
                                     min_freq)
            if token_to_idx:
                self._sort_index_according_to_user_specification(token_to_idx)
            if unk_token:
                self._token_to_idx.default_factory = lambda: self._token_to_idx[
                    unk_token]

        # _expose_tokens_as_attributes
        self._identifiers_to_tokens = kwargs
        for identifier, token in kwargs.items():
            if identifier.startswith('_'):
                raise ValueError(
                    'It is not allowed to use identifiers starting with '
                    'underscore. In Python identifier names beginning with '
                    'underscore are internal.')
            if hasattr(self, identifier):
                raise ValueError(
                    'vocab.{} already exists. '
                    'Please choose a different identifier for token {}'.format(
                        identifier, token))
            setattr(self, identifier, token)

    def _index_counter_keys(self, counter, special_tokens, max_size, min_freq):
        # sort by frequency, then alphabetically
        token_freqs = sorted(counter.items(), key=lambda x: x[0])
        token_freqs.sort(key=lambda x: x[1], reverse=True)
        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        special_tokens = set(special_tokens)
        max_size = None if max_size is None else max_size + len(special_tokens)
        for token, freq in token_freqs:
            if freq < min_freq or len(self._idx_to_token) == max_size:
                break
            if token not in special_tokens:
                self._idx_to_token[max(list(self._idx_to_token.keys()) + [-1]) +
                                   1] = token
                self._token_to_idx[token] = max(self._idx_to_token.keys())

    def _sort_index_according_to_user_specification(self, token_to_idx):
        # Sanity checks
        if not set(token_to_idx.keys()).issubset(self.token_to_idx.keys()):
            raise ValueError(
                'User-specified token_to_idx mapping can only contain '
                'tokens that will be part of the vocabulary.')
        if len(set(token_to_idx.values())) != len(token_to_idx):
            raise ValueError(
                'User-specified indices must not contain duplicates.')
        if min(token_to_idx.values()) < 0 or max(token_to_idx.values()) >= len(
                self.token_to_idx):
            raise ValueError(
                'User-specified indices must not be < 0 or >= the number of tokens '
                'that will be in the vocabulary. The current vocab contains {}'
                'tokens.'.format(len(self.token_to_idx)))

        # Update index ordering
        for token, new_idx in token_to_idx.items():
            old_idx = self.token_to_idx[token]
            ousted_token = self.idx_to_token[new_idx]

            self.token_to_idx[token] = new_idx
            self.token_to_idx[ousted_token] = old_idx
            self.idx_to_token[old_idx] = ousted_token
            self.idx_to_token[new_idx] = token

    def to_tokens(self, indices):
        """
        Maps the input indices to token list.

        Args:
            indices (int|list[int]|tuple[int]|numpy.ndarray): The input indice(s) for mapping.
                Must be an `int` or 1D `list[int]`|`tuple[int]`|`numpy.ndarray`.

        Returns:
            str|list[str]: Obtained token(s). If `indices` is an integer, it 
            will return a str. If `indices` is a list/tuple of integers, it will 
            return a list of str.
        """
        to_reduce = False
        if not isinstance(indices, (list, tuple, np.ndarray)):
            indices = [indices]
            to_reduce = True
        if isinstance(indices, (list, tuple)):
            indices = np.asarray(indices)

        if isinstance(indices, (np.ndarray)) and len(indices.shape) > 1:
            raise ValueError(
                'Token indices is invalid. Expected 1D array, but received {}D array. '
                .format(len(indices.shape)))

        tokens = []
        for idx in indices:
            if not isinstance(idx, (int, np.integer)):
                warnings.warn(
                    "The type of `to_tokens()`'s input `indices` is not `int` which will be forcibly transfered to `int`. "
                )
                idx = int(idx)

            try:
                tokens.append(self._idx_to_token[idx])
            except KeyError:
                raise ValueError(
                    'Token index {} in the provided `indices` is invalid.'.
                    format(idx))

        return tokens[0] if to_reduce else tokens

    def to_indices(self, tokens):
        """
        Maps the input tokens into indices.

        Args:
            tokens (str|list[str]|tuple[str], optional): The input token(s) for 
                mapping.
        
        Returns:
            int|list[int]: Obationed indice(s). If `tokens` is a str, it will 
            return an integer. If `tokens` is a list/tuple of str, it will 
            return a list of integers.
        """
        return self[tokens]

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self._token_to_idx[tokens]
        else:
            return [self._token_to_idx[token] for token in tokens]

    def __len__(self):
        return len(self._idx_to_token)

    def __contains__(self, token):
        return token in self._token_to_idx

    def __call__(self, tokens):
        """
        Maps the input tokens into indices. Its function is the same as the 
        :meth:`to_indices` method.

        See detail at `to_indices`.
        """
        return self[tokens]

    @property
    def idx_to_token(self):
        # Returns index-token dict
        return self._idx_to_token

    @property
    def token_to_idx(self):
        # Return token-index dict
        return self._token_to_idx

    def to_json(self, path=None):
        """
        Summarizes some information of vocab as JSON string. If path is gaven,
        the JSON string will be saved into files. The JSON string and the saved
        file all can be used to reconstruct the :class:`Vocab` by calling 
        :meth:`from_json` method.

        Args:
            path (str, optional): The path to save JSON string. If None, the
                JSON will not be saved. Default: None.
        
        Returns:
            str: The JSON string including information of vocab.
        """
        vocab_dict = {}
        vocab_dict['idx_to_token'] = dict(self.idx_to_token)
        vocab_dict['token_to_idx'] = dict(self.token_to_idx)
        vocab_dict['unk_token'] = self.unk_token
        vocab_dict['identifiers_to_tokens'] = self._identifiers_to_tokens
        json_str = json.dumps(vocab_dict)
        if path:
            with io.open(path, 'w', encoding='utf-8') as f:
                f.write(json_str)
        return json_str

    @classmethod
    def from_json(cls, json_str):
        """
        Loads :class:`Vocab` from JSON string or JSON file, which is gotten by 
        calling :meth:`to_json` method.

        Args:
            json_str (str): JSON string or file path of JSON string.

        Returns:
            Vocab: An instance of :class:`Vocab` generated from information 
            contained in JSON string.
        """
        if os.path.isfile(json_str):
            with io.open(json_str, 'r', encoding='utf-8') as f:
                vocab_dict = json.load(f)
        else:
            vocab_dict = json.loads(json_str)
        token_to_idx = vocab_dict.get('token_to_idx')
        unk_token = vocab_dict.get('unk_token')
        identifiers_to_tokens = vocab_dict.get('identifiers_to_tokens', dict())
        if 'unk_token' in identifiers_to_tokens:
            del identifiers_to_tokens['unk_token']
        vocab = cls(counter=None,
                    token_to_idx=token_to_idx,
                    unk_token=unk_token,
                    **identifiers_to_tokens)
        return vocab

    @classmethod
    def from_dict(cls,
                  token_to_idx,
                  unk_token=None,
                  pad_token=None,
                  bos_token=None,
                  eos_token=None,
                  **kwargs):
        """
        Builds the :class:`Vocab` from a dict.
        """
        vocab = cls(counter=None,
                    token_to_idx=token_to_idx,
                    unk_token=unk_token,
                    pad_token=pad_token,
                    bos_token=bos_token,
                    eos_token=eos_token,
                    **kwargs)
        return vocab

    @staticmethod
    def build_vocab(iterator,
                    max_size=None,
                    min_freq=1,
                    token_to_idx=None,
                    unk_token=None,
                    pad_token=None,
                    bos_token=None,
                    eos_token=None,
                    **kwargs):
        """
        Builds the :class:`Vocab` accoring to given iterator and other 
        information. Firstly, iterate over the `iterator` to construct a 
        :class:`collections.Counter` and used to init the as  :class:`Vocab`.
        """
        counter = collections.Counter()
        for tokens in iterator:
            counter.update(tokens)
        vocab = Vocab(counter,
                      max_size=max_size,
                      min_freq=min_freq,
                      token_to_idx=token_to_idx,
                      unk_token=unk_token,
                      pad_token=pad_token,
                      bos_token=bos_token,
                      eos_token=eos_token,
                      **kwargs)
        return vocab

    @staticmethod
    def load_vocabulary(filepath,
                        unk_token=None,
                        pad_token=None,
                        bos_token=None,
                        eos_token=None,
                        **kwargs):
        """
        Builds the :class:`Vocab` from a file reserving all tokens by calling 
        :meth:`Vocab.from_dict` method. The file contains a token per line, and 
        the line index would be the index of corresponding token.
        """
        token_to_idx = {}
        with io.open(filepath, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f):
                token = line.rstrip('\n')
                token_to_idx[token] = int(index)
        vocab = Vocab.from_dict(token_to_idx,
                                unk_token=unk_token,
                                pad_token=pad_token,
                                bos_token=bos_token,
                                eos_token=eos_token,
                                **kwargs)
        return vocab


class SpecialTokensMixin:
    """
    A mixin derived by [`PretrainedTokenizer`] to handle specific behaviors related to
    special tokens. In particular, this class hold the attributes which can be used to directly access these special
    tokens in a model-independent manner and allow to set and update the special tokens.
    """

    SPECIAL_TOKENS_ATTRIBUTES = [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
        "additional_special_tokens",
    ]

    def __init__(self, do_lower_case=True, verbose=True, **kwargs):
        
        self._bos_token = builtins.getattr(self, "_bos_token", None)
        self._eos_token = getattr(self, "_eos_token", None)
        self._unk_token = getattr(self, "_unk_token", None)
        self._sep_token = getattr(self, "_sep_token", None)
        self._pad_token = getattr(self, "_pad_token", None)
        self._cls_token = getattr(self, "_cls_token", None)
        self._mask_token = getattr(self, "_mask_token", None)
        self._pad_token_type_id = getattr(self, "_pad_token_type_id", 0)
        self._additional_special_tokens = getattr(self,
                                                  "_additional_special_tokens",
                                                  [])
        self.verbose = verbose,
        self._unk_token =  "[UNK]"
        self._sep_token =  "[SEP]"
        self._pad_token =  "[PAD]"
        self._cls_token =  "[CLS]"
        self._mask_token =  "[MASK]"
        self.do_lower_case = do_lower_case

        self.added_tokens_encoder: Dict[str, int] = {}
        self.added_tokens_decoder: Dict[int, str] = {}
        self.unique_no_split_tokens: List[str] = ['[CLS]', '[MASK]', '[PAD]', '[SEP]', '[UNK]']
        self.tokens_trie = Trie()
        self._decode_use_source_tokenizer = False

    def sanitize_special_tokens(self) -> int:
        """
        Make sure that all the special tokens attributes of the tokenizer (`tokenizer.mask_token`,
        `tokenizer.cls_token`, etc.) are in the vocabulary.

        Add the missing ones to the vocabulary if needed.

        Return:
            `int`: The number of tokens added in the vocabulary during the operation.
        """
        return self.add_tokens(self.all_special_tokens_extended,
                               special_tokens=True)
    
    def _create_trie(self, unique_no_split_tokens):
        trie = Trie()
        for token in unique_no_split_tokens:
            if hasattr(
                    self, "do_lower_case"
            ) and self.do_lower_case and token not in self.all_special_tokens:
                trie.add(token.lower())
            else:
                trie.add(token)
        self.tokens_trie = trie

    def add_special_tokens(
            self, special_tokens_dict: Dict[str, Union[str,
                                                       AddedToken]]) -> int:
        """
        Add a dictionary of special tokens (eos, pad, cls, etc.) to the encoder and link them to class attributes. If
        special tokens are NOT in the vocabulary, they are added to it (indexed starting from the last index of the
        current vocabulary).
        """
        if not special_tokens_dict:
            return 0

        added_tokens = 0
        for key, value in special_tokens_dict.items():
            assert key in self.SPECIAL_TOKENS_ATTRIBUTES, f"Key {key} is not a special token"

            setattr(self, key, value)

            if key == "additional_special_tokens":
                assert isinstance(value, (list, tuple)) and all(
                    isinstance(t, (str, AddedToken)) for t in value
                ), f"Tokens {value} for key {key} should all be str or AddedToken instances"
                added_tokens += self.add_tokens(value, special_tokens=True)
            else:
                assert isinstance(
                    value, (str, AddedToken)
                ), f"Token {value} for key {key} should be a str or an AddedToken instance"
                added_tokens += self.add_tokens([value], special_tokens=True)

        return added_tokens

    def add_tokens(self,
                   new_tokens: Union[str, AddedToken, List[Union[str,
                                                                 AddedToken]]],
                   special_tokens: bool = False) -> int:
        """
        Add a list of new tokens to the tokenizer class. If the new tokens are not in the vocabulary, they are added to
        it with indices starting from length of the current vocabulary.

        Note,None When adding new tokens to the vocabulary, you should make sure to also resize the token embedding
        matrix of the model so that its embedding matrix matches the tokenizer.

        In order to do that, please use the [`~PreTrainedModel.resize_token_embeddings`] method.
        """
        if not new_tokens:
            return 0

        if not isinstance(new_tokens, (list, tuple)):
            new_tokens = [new_tokens]

        return self._add_tokens(new_tokens, special_tokens=special_tokens)

    def _add_tokens(self,
                    new_tokens: Union[List[str], List[AddedToken]],
                    special_tokens: bool = False) -> int:
        new_tokens = [str(tok) for tok in new_tokens]

        tokens_to_add = []
        for token in new_tokens:
            if not isinstance(token, str):
                raise TypeError(
                    f"Token {token} is not a string but a {type(token)}.")
            if not special_tokens and hasattr(
                    self, "do_lower_case") and self.do_lower_case:
                token = token.lower()
            if (token != self.unk_token and self.convert_tokens_to_ids(token)
                    == self.convert_tokens_to_ids(self.unk_token)
                    and token not in tokens_to_add):
                tokens_to_add.append(token)
                if self.verbose:
                    pass

        added_tok_encoder = dict(
            (tok, len(self) + i) for i, tok in enumerate(tokens_to_add))
        added_tok_decoder = {v: k for k, v in added_tok_encoder.items()}
        self.added_tokens_encoder.update(added_tok_encoder)
        self.added_tokens_decoder.update(added_tok_decoder)

        # Make sure we don't split on any special tokens (even they were already in the vocab before e.g. for Albert)
        if special_tokens:
            if len(new_tokens) == 1:
                _insert_one_token_to_ordered_list(self.unique_no_split_tokens,
                                                  new_tokens[0])
            else:
                self.unique_no_split_tokens = sorted(
                    set(self.unique_no_split_tokens).union(set(new_tokens)))
        else:
            # Or on the newly added tokens
            if len(tokens_to_add) == 1:
                _insert_one_token_to_ordered_list(self.unique_no_split_tokens,
                                                  tokens_to_add[0])
            else:
                self.unique_no_split_tokens = sorted(
                    set(self.unique_no_split_tokens).union(set(tokens_to_add)))
        self._create_trie(self.unique_no_split_tokens)

        return len(tokens_to_add)

    @property
    def bos_token(self) -> str:
        """
        `str`: Beginning of sentence token. Log an error if used while not having been set.
        """
        if self._bos_token is None and self.verbose:
            return None
        return str(self._bos_token)

    @property
    def eos_token(self) -> str:
        """
        `str`: End of sentence token. Log an error if used while not having been set.
        """
        if self._eos_token is None and self.verbose:
            return None
        return str(self._eos_token)

    @property
    def unk_token(self) -> str:
        """
        `str`: Unknown token. Log an error if used while not having been set.
        """
        if self._unk_token is None and self.verbose:
            return None
        return str(self._unk_token)

    @property
    def sep_token(self) -> str:
        """
        `str`: Separation token, to separate context and query in an input sequence. Log an error if used while not
        having been set.
        """
        if self._sep_token is None and self.verbose:
            return None
        return str(self._sep_token)

    @property
    def pad_token(self) -> str:
        """
        `str`: Padding token. Log an error if used while not having been set.
        """
        if self._pad_token is None and self.verbose:
            # logger.error("Using pad_token, but it is not set yet.")
            return None
        return str(self._pad_token)

    @property
    def cls_token(self) -> str:
        """
        `str`: Classification token, to extract a summary of an input sequence leveraging self-attention along the full
        depth of the model. Log an error if used while not having been set.
        """
        if self._cls_token is None and self.verbose:
            return None
        return str(self._cls_token)

    @property
    def mask_token(self) -> str:
        """
        `str`: Mask token, to use when training a model with masked-language modeling. Log an error if used while not
        having been set.
        """
        if self._mask_token is None and self.verbose:
            return None
        return str(self._mask_token)

    @property
    def additional_special_tokens(self) -> List[str]:
        """
        `List[str]`: All the additional special tokens you may want to use. Log an error if used while not having been
        set.
        """
        if self._additional_special_tokens is None and self.verbose:
            return None
        return [str(tok) for tok in self._additional_special_tokens]

    @bos_token.setter
    def bos_token(self, value):
        self._bos_token = value

    @eos_token.setter
    def eos_token(self, value):
        self._eos_token = value

    @unk_token.setter
    def unk_token(self, value):
        self._unk_token = value

    @sep_token.setter
    def sep_token(self, value):
        self._sep_token = value

    @pad_token.setter
    def pad_token(self, value):
        self._pad_token = value

    @cls_token.setter
    def cls_token(self, value):
        self._cls_token = value

    @mask_token.setter
    def mask_token(self, value):
        self._mask_token = value

    @additional_special_tokens.setter
    def additional_special_tokens(self, value):
        self._additional_special_tokens = value

    @property
    def bos_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the beginning of sentence token in the vocabulary. Returns `None` if the token has not
        been set.
        """
        if self._bos_token is None:
            return None
        return self.convert_tokens_to_ids(self.bos_token)

    @property
    def eos_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the end of sentence token in the vocabulary. Returns `None` if the token has not been
        set.
        """
        if self._eos_token is None:
            return None
        return self.convert_tokens_to_ids(self.eos_token)

    @property
    def unk_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the unknown token in the vocabulary. Returns `None` if the token has not been set.
        """
        if self._unk_token is None:
            return None
        return self.convert_tokens_to_ids(self.unk_token)

    @property
    def sep_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the separation token in the vocabulary, to separate context and query in an input
        sequence. Returns `None` if the token has not been set.
        """
        if self._sep_token is None:
            return None
        return self.convert_tokens_to_ids(self.sep_token)

    @property
    def pad_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the padding token in the vocabulary. Returns `None` if the token has not been set.
        """
        if self._pad_token is None:
            return None
        return self.convert_tokens_to_ids(self.pad_token)

    @property
    def pad_token_type_id(self) -> int:
        """
        `int`: Id of the padding token type in the vocabulary.
        """
        return self._pad_token_type_id

    @property
    def cls_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the classification token in the vocabulary, to extract a summary of an input sequence
        leveraging self-attention along the full depth of the model.

        Returns `None` if the token has not been set.
        """
        if self._cls_token is None:
            return None
        return self.convert_tokens_to_ids(self.cls_token)

    @property
    def mask_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the mask token in the vocabulary, used when training a model with masked-language
        modeling. Returns `None` if the token has not been set.
        """
        if self._mask_token is None:
            return None
        return self.convert_tokens_to_ids(self.mask_token)

    @property
    def additional_special_tokens_ids(self) -> List[int]:
        """
        `List[int]`: Ids of all the additional special tokens in the vocabulary. Log an error if used while not having
        been set.
        """
        return self.convert_tokens_to_ids(self.additional_special_tokens)

    @bos_token_id.setter
    def bos_token_id(self, value):
        self._bos_token = self.convert_ids_to_tokens(
            value) if value is not None else None

    @eos_token_id.setter
    def eos_token_id(self, value):
        self._eos_token = self.convert_ids_to_tokens(
            value) if value is not None else None

    @unk_token_id.setter
    def unk_token_id(self, value):
        self._unk_token = self.convert_ids_to_tokens(
            value) if value is not None else None

    @sep_token_id.setter
    def sep_token_id(self, value):
        self._sep_token = self.convert_ids_to_tokens(
            value) if value is not None else None

    @pad_token_id.setter
    def pad_token_id(self, value):
        self._pad_token = self.convert_ids_to_tokens(
            value) if value is not None else None

    @cls_token_id.setter
    def cls_token_id(self, value):
        self._cls_token = self.convert_ids_to_tokens(
            value) if value is not None else None

    @mask_token_id.setter
    def mask_token_id(self, value):
        self._mask_token = self.convert_ids_to_tokens(
            value) if value is not None else None

    @additional_special_tokens_ids.setter
    def additional_special_tokens_ids(self, values):
        self._additional_special_tokens = [
            self.convert_ids_to_tokens(value) for value in values
        ]

    @property
    def special_tokens_map(self) -> Dict[str, Union[str, List[str]]]:
        """
        `Dict[str, Union[str, List[str]]]`: A dictionary mapping special token class attributes (`cls_token`,
        `unk_token`, etc.) to their values (`'<unk>'`, `'<cls>'`, etc.).

        Convert potential tokens of `AddedToken` type to string.
        """
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, "_" + attr)
            if attr_value:
                set_attr[attr] = (type(attr_value)(
                    str(attr_value_sub)
                    for attr_value_sub in attr_value) if isinstance(
                        attr_value, (list, tuple)) else str(attr_value))
        return set_attr

    @property
    def special_tokens_map_extended(
        self
    ) -> Dict[str, Union[str, AddedToken, List[Union[str, AddedToken]]]]:
        """
        `Dict[str, Union[str, AddedToken, List[Union[str, AddedToken]]]]`: A dictionary mapping
        special token class attributes (`cls_token`, `unk_token`, etc.) to their values (`'<unk>'`, `'<cls>'`, etc.).

        Don't convert tokens of `AddedToken` type to string so they can be used to control more finely how
        special tokens are tokenized.
        """
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, "_" + attr)
            if attr_value:
                set_attr[attr] = attr_value
        return set_attr

    @property
    def all_special_tokens(self) -> List[str]:
        """
        `List[str]`: All the special tokens (`'<unk>'`, `'<cls>'`, etc.) mapped to class attributes.

        Convert tokens of `AddedToken` type to string.
        """
        all_toks = [str(s) for s in self.all_special_tokens_extended]
        return all_toks

    @property
    def all_special_tokens_extended(self) -> List[Union[str, AddedToken]]:
        """
        `List[Union[str, AddedToken]]`: All the special tokens (`'<unk>'`, `'<cls>'`, etc.) mapped to class
        attributes.

        Don't convert tokens of `AddedToken` type to string so they can be used to control more finely how
        special tokens are tokenized.
        """
        all_toks = []
        set_attr = self.special_tokens_map_extended
        for attr_value in set_attr.values():
             all_toks = all_toks + (list(attr_value) if isinstance(
                attr_value, (list, tuple)) else [attr_value])
        all_toks = list(OrderedDict.fromkeys(all_toks))
        return all_toks

    @property
    def all_special_ids(self) -> List[int]:
        """
        `List[int]`: List the ids of the special tokens(`'<unk>'`, `'<cls>'`, etc.) mapped to class attributes.
        """
        all_toks = self.all_special_tokens
        all_ids = self.convert_tokens_to_ids(all_toks)
        return all_ids
    
    def convert_tokens_to_ids(self, tokens):
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))

        return ids
    
    def _convert_token_to_id_with_added_voc(self, token):
        if token is None:
            return None

        if token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]
        return self._convert_token_to_id(token)

    def _convert_token_to_id(self, token):

        return self.vocab.to_indices(token)


class BasicTokenizer(object):
    """
    Runs basic tokenization (punctuation splitting, lower casing, etc.).

    Args:
        do_lower_case (bool):
            Whether or not to lowercase the input when tokenizing.
            Defaults to `True`.

    """

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer."""
        super().__init__()

        self.do_lower_case = do_lower_case
        # self.additional_special_tokens = []

    def tokenize(self, text):
        """
        Tokenizes a piece of text using basic tokenizer.

        Args:
            text (str): A piece of text.

        Returns: 
            list(str): A list of tokens.
        """

        text = convert_to_unicode(text)
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            else:
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """
        Strips accents from a piece of text.
        """
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """
        Splits punctuation on a piece of text.
        """
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """
        Adds whitespace around any CJK character.
        """
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """
        Checks whether CP is the codepoint of a CJK character.
        """
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """
        Performs invalid character removal and whitespace cleanup on text.
        """
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """
    Runs WordPiece tokenization.

    Args:
        vocab (Vocab|dict):
            Vocab of the word piece tokenizer.
        unk_token (str):
            A specific token to replace all unknown tokens.
        max_input_chars_per_word (int):
            If a word's length is more than
            max_input_chars_per_word, it will be dealt as unknown word.
            Defaults to 100.
    """

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces.
        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


class MyTokenizer(SpecialTokensMixin):
    """Tokenizer"""

    def __init__(self, vocab_file, 
                split_char=" ", 
                unk_token="[UNK]", 
                sep_token="[SEP]",
                pad_token="[PAD]",
                cls_token="[CLS]",
                mask_token="[MASK]",
                do_lower_case=True, 
                params=None, 
                **kwargs):
        """
        :param vocab_file: 词表文件路径
        :param split_char: 明文分隔符，默认是空格
        :param unk_token: unk 对应的token，默认是[UNK]
        :param params: 个别tokenizer自己用到的额外参数，dict类型
        """
        # self.vocabulary = Vocabulary(vocab_file, unk_token)
        super(MyTokenizer, self).__init__(SpecialTokensMixin)

        self.vocab = Vocab.load_vocabulary(vocab_file, unk_token=unk_token)
        self.split_char = split_char
        self.unk_token = unk_token
        self.params = params
        self.do_lower_case = do_lower_case
        self.padding_side = 'right'
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab,
                                                      unk_token=unk_token)

        model_max_length = kwargs.pop("model_max_length",
                                      kwargs.pop("max_len", None))
        self.model_max_length = model_max_length if model_max_length is not None else VERY_LARGE_INTEGER
        self.model_input_names = ["input_ids", "token_type_ids"]
        self.truncation_side = "right"
        self.slow_tokenizer_class = None

        # Padding and truncation side are right by default and overridden in subclasses. If specified in the kwargs, it
        # is changed.
        self.padding_side = kwargs.pop("padding_side", self.padding_side)
        if self.padding_side not in ["right", "left"]:
            raise ValueError(
                f"Padding side should be selected between 'right' and 'left', current value: {self.padding_side}"
            )

        self.truncation_side = kwargs.pop("truncation_side", self.truncation_side)

        if self.truncation_side not in ["right", "left"]:
            raise ValueError(
                f"Padding side should be selected between 'right' and 'left', current value: {self.truncation_side}"
            )

        self.model_input_names = kwargs.pop("model_input_names",
                                            self.model_input_names)

        self.deprecation_warnings = (
            {}
        )

        # self.added_tokens_encoder: Dict[str, int] = {}
        # self.added_tokens_decoder: Dict[int, str] = {}
        # self.unique_no_split_tokens: List[str] = [] 
        # self.tokens_trie = Trie()
        # self._decode_use_source_tokenizer = False









    
    def _get_padding_truncation_strategies(self,
                                        padding=False,
                                        truncation=False,
                                        max_length=None,
                                        pad_to_multiple_of=None,
                                        verbose=True,
                                        **kwargs):
        """
        Find the correct padding/truncation strategy with backward compatibility for old arguments (truncation_strategy
        and pad_to_max_length) and behaviors.
        """
        old_truncation_strategy = kwargs.pop("truncation_strategy",
                                             "do_not_truncate")
        old_pad_to_max_length = kwargs.pop("pad_to_max_seq_len", False)

        # Backward compatibility for previous behavior, maybe we should deprecate it:
        # If you only set max_length, it activates truncation for max_length
        if max_length is not None and padding is False and truncation is False:
            if verbose:
                if not self.deprecation_warnings.get(
                        "Truncation-not-explicitly-activated", False):
                    warnings.warn(
                        "Truncation was not explicitly activated but `max_length` is provided a specific value, "
                        "please use `truncation=True` to explicitly truncate examples to max length. "
                        "Defaulting to 'longest_first' truncation strategy. "
                        "If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy "
                        "more precisely by providing a specific strategy to `truncation`."
                    )
                self.deprecation_warnings[
                    "Truncation-not-explicitly-activated"] = True
            truncation = "longest_first"

        # Get padding strategy
        if padding is False and old_pad_to_max_length:
            if verbose:
                warnings.warn(
                    "The `pad_to_max_length` argument is deprecated and will be removed in a future version, "
                    "use `padding=True` or `padding='longest'` to pad to the longest sequence in the batch, or "
                    "use `padding='max_length'` to pad to a max length. In this case, you can give a specific "
                    "length with `max_length` (e.g. `max_length=45`) or leave max_length to None to pad to the "
                    "maximal input size of the model (e.g. 512 for Bert).",
                    FutureWarning,
                )
            if max_length is None:
                padding_strategy = PaddingStrategy.LONGEST
            else:
                padding_strategy = PaddingStrategy.MAX_LENGTH
        elif padding is not False:
            if padding is True:
                if verbose:
                    if max_length is not None and (truncation is False
                                                   or truncation
                                                   == "do_not_truncate"):
                        warnings.warn(
                            "`max_length` is ignored when `padding`=`True` and there is no truncation strategy. "
                            "To pad to max length, use `padding='max_length'`.")
                    if old_pad_to_max_length is not False:
                        warnings.warn(
                            "Though `pad_to_max_length` = `True`, it is ignored because `padding`=`True`."
                        )
                padding_strategy = PaddingStrategy.LONGEST  # Default to pad to the longest sequence in the batch
            elif not isinstance(padding, PaddingStrategy):
                padding_strategy = PaddingStrategy(padding)
            elif isinstance(padding, PaddingStrategy):
                padding_strategy = padding
        else:
            padding_strategy = PaddingStrategy.DO_NOT_PAD

        # Get truncation strategy
        if truncation is False and old_truncation_strategy != "do_not_truncate":
            if verbose:
                warnings.warn(
                    "The `truncation_strategy` argument is deprecated and will be removed in a future version, "
                    "use `truncation=True` to truncate examples to a max length. You can give a specific "
                    "length with `max_length` (e.g. `max_length=45`) or leave max_length to None to truncate to the "
                    "maximal input size of the model (e.g. 512 for Bert). "
                    " If you have pairs of inputs, you can give a specific truncation strategy selected among "
                    "`truncation='only_first'` (will only truncate the first sentence in the pairs) "
                    "`truncation='only_second'` (will only truncate the second sentence in the pairs) "
                    "or `truncation='longest_first'` (will iteratively remove tokens from the longest sentence in the pairs).",
                    FutureWarning,
                )
            truncation_strategy = TruncationStrategy(old_truncation_strategy)
        elif truncation is not False:
            if truncation is True:
                truncation_strategy = (
                    TruncationStrategy.LONGEST_FIRST
                )  # Default to truncate the longest sequences in pairs of inputs
            elif not isinstance(truncation, TruncationStrategy):
                truncation_strategy = TruncationStrategy(truncation)
            elif isinstance(truncation, TruncationStrategy):
                truncation_strategy = truncation
        else:
            truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE

        # Set max length if needed
        if max_length is None:
            if padding_strategy == PaddingStrategy.MAX_LENGTH:
                if self.model_max_length > LARGE_INTEGER:
                    if verbose:
                        if not self.deprecation_warnings.get(
                                "Asking-to-pad-to-max_length", False):
                            warnings.warn(
                                "Asking to pad to max_length but no maximum length is provided and the model has no predefined maximum length. "
                                "Default to no padding.")
                        self.deprecation_warnings[
                            "Asking-to-pad-to-max_length"] = True
                    padding_strategy = PaddingStrategy.DO_NOT_PAD
                else:
                    max_length = self.model_max_length

            if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE:
                if self.model_max_length > LARGE_INTEGER:
                    if verbose:
                        if not self.deprecation_warnings.get(
                                "Asking-to-truncate-to-max_length", False):
                            warnings.warn(
                                "Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. "
                                "Default to no truncation.")
                        self.deprecation_warnings[
                            "Asking-to-truncate-to-max_length"] = True
                    truncation_strategy = TruncationStrategy.DO_NOT_TRUNCATE
                else:
                    max_length = self.model_max_length

        # Test if we have a padding token
        if padding_strategy != PaddingStrategy.DO_NOT_PAD and (
                not self.pad_token or self.pad_token_id < 0):
            raise ValueError(
                "Asking to pad but the tokenizer does not have a padding token. "
                "Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` "
                "or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`."
            )

        # Check that we will truncate to a multiple of pad_to_multiple_of if both are provided
        if (truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE
                and padding_strategy != PaddingStrategy.DO_NOT_PAD
                and pad_to_multiple_of is not None and max_length is not None
                and (max_length % pad_to_multiple_of != 0)):
            raise ValueError(
                f"Truncation and padding are both activated but "
                f"truncation length ({max_length}) is not a multiple of pad_to_multiple_of ({pad_to_multiple_of})."
            )

        return padding_strategy, truncation_strategy, max_length, kwargs

    
    def __call__(self,
                text,
                text_pair=None,
                max_length: Optional[int] = None,
                stride=0,
                is_split_into_words=False,
                padding: Union[bool, str, PaddingStrategy] = False,
                truncation: Union[bool, str, TruncationStrategy] = False,
                return_position_ids=False,
                return_token_type_ids=True,
                return_attention_mask=False,
                return_length=False,
                return_overflowing_tokens=False,
                return_special_tokens_mask=False,
                return_dict=True,
                return_offsets_mapping=False,
                add_special_tokens=True,
                pad_to_multiple_of: Optional[int] = None,
                return_tensors: Optional[Union[str, TensorType]] = None,
                verbose: bool = True,
                **kwargs):
  

        # Input type checking for clearer error
        def _is_valid_text_input(t):
            if isinstance(t, str):
                # Strings are fine
                return True
            elif isinstance(t, (list, tuple)):
                # List are fine as long as they are...
                if len(t) == 0:
                    # ... empty
                    return True
                elif isinstance(t[0], str):
                    # ... list of strings
                    return True
                elif isinstance(t[0], (list, tuple)):
                    # ... list with an empty list or with a list of strings
                    return len(t[0]) == 0 or isinstance(t[0][0], str)
                else:
                    return False
            else:
                return False

        if not _is_valid_text_input(text):
            raise ValueError(
                "text input must of type `str` (single example), `List[str]` (batch or single pretokenized example) "
                "or `List[List[str]]` (batch of pretokenized examples).")

        if not _is_valid_text_input(text):
            raise ValueError(
                "text input must of type `str` (single example), `List[str]` (batch or single pretokenized example) "
                "or `List[List[str]]` (batch of pretokenized examples).")

        if text_pair is not None and not _is_valid_text_input(text_pair):
            raise ValueError(
                "text input must of type `str` (single example), `List[str]` (batch or single pretokenized example) "
                "or `List[List[str]]` (batch of pretokenized examples).")

        if is_split_into_words:
            is_batched = isinstance(text,
                                    (list, tuple)) and text and isinstance(
                                        text[0], (list, tuple))
        else:
            is_batched = isinstance(text, (list, tuple))

        if is_batched:
            if isinstance(text_pair, str):
                raise TypeError(
                    "when tokenizing batches of text, `text_pair` must be a list or tuple with the same length as `text`."
                )
            if text_pair is not None and len(text) != len(text_pair):
                raise ValueError(
                    f"batch length of `text`: {len(text)} does not match batch length of `text_pair`: {len(text_pair)}."
                )
            batch_text_or_text_pairs = list(zip(
                text, text_pair)) if text_pair is not None else text
            return self.batch_encode(
                batch_text_or_text_pairs=batch_text_or_text_pairs,
                max_length=max_length,
                stride=stride,
                is_split_into_words=is_split_into_words,
                padding=padding,
                truncation=truncation,
                return_position_ids=return_position_ids,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_length=return_length,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_dict=return_dict,
                return_offsets_mapping=return_offsets_mapping,
                add_special_tokens=add_special_tokens,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors=return_tensors,
                verbose=verbose,
                **kwargs)
        else:
            return self.encode(
                text=text,
                text_pair=text_pair,
                max_length=max_length,
                stride=stride,
                is_split_into_words=is_split_into_words,
                padding=padding,
                truncation=truncation,
                return_position_ids=return_position_ids,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_length=return_length,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                add_special_tokens=add_special_tokens,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors=return_tensors,
                verbose=verbose,
                **kwargs)
    
    def encode(self,
            text,
            text_pair=None,
            max_length=None,
            stride: int = 0,
            is_split_into_words: bool = False,
            padding: Union[bool, str, PaddingStrategy] = False,
            truncation: Union[bool, str, TruncationStrategy] = False,
            return_position_ids=False,
            return_token_type_ids=True,
            return_attention_mask=False,
            return_length=False,
            return_overflowing_tokens=False,
            return_special_tokens_mask=False,
            return_offsets_mapping=False,
            add_special_tokens=True,
            pad_to_multiple_of: Optional[int] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            verbose: bool = True,
            **kwargs):
        """
        Tokenize and prepare for the model a sequence or a pair of sequences.
        """
        # Backward compatibility for 'max_seq_len'
        old_max_seq_len = kwargs.get('max_seq_len', None)
        if max_length is None and old_max_seq_len:
            if verbose:
                warnings.warn(
                    "The `max_seq_len` argument is deprecated and will be removed in a future version, "
                    "please use `max_length` instead.",
                    FutureWarning,
                )
            max_length = old_max_seq_len
        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        return self._encode_plus(
            text=text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_position_ids=return_position_ids,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )
    
    def batch_encode(self,
                    batch_text_or_text_pairs: Union[
                        List[TextInput], List[TextInputPair],
                        List[PreTokenizedInput], List[PreTokenizedInputPair],
                        List[EncodedInput], List[EncodedInputPair], ],
                    max_length=None,
                    stride: int = 0,
                    is_split_into_words: bool = False,
                    padding: Union[bool, str, PaddingStrategy] = False,
                    truncation: Union[bool, str, TruncationStrategy] = False,
                    return_position_ids=False,
                    return_token_type_ids=True,
                    return_attention_mask=False,
                    return_length=False,
                    return_overflowing_tokens=False,
                    return_special_tokens_mask=False,
                    return_dict=True,
                    return_offsets_mapping=False,
                    add_special_tokens=True,
                    pad_to_multiple_of: Optional[int] = None,
                    return_tensors: Optional[Union[str, TensorType]] = None,
                    verbose: bool = True,
                    **kwargs):

        # Backward compatibility for 'max_seq_len'
        old_max_seq_len = kwargs.get('max_seq_len', None)
        if max_length is None and old_max_seq_len:
            if verbose:
                warnings.warn(
                    "The `max_seq_len` argument is deprecated and will be removed in a future version, "
                    "please use `max_length` instead.",
                    FutureWarning,
                )
            max_length = old_max_seq_len
        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        return self._batch_encode_plus(
            batch_text_or_text_pairs=batch_text_or_text_pairs,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_position_ids=return_position_ids,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_dict=return_dict,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )
    

    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[List[TextInput],
                                        List[TextInputPair],
                                        List[PreTokenizedInput],
                                        List[PreTokenizedInputPair],
                                        List[EncodedInput],
                                        List[EncodedInputPair], ],
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_position_ids: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_dict: bool = True,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs):

        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text, **kwargs)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text,
                            (list, tuple)) and len(text) > 0 and isinstance(
                                text[0], str):
                #TODO aligns with HuggingFace here in breaking change
                return self.convert_tokens_to_ids(text)
            elif isinstance(text,
                            (list, tuple)) and len(text) > 0 and isinstance(
                                text[0], int):
                return text
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        input_ids = []
        for ids_or_pair_ids in batch_text_or_text_pairs:
            if not isinstance(ids_or_pair_ids, (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            elif is_split_into_words and not isinstance(ids_or_pair_ids[0],
                                                        (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            else:
                ids, pair_ids = ids_or_pair_ids

            first_ids = get_input_ids(ids)
            second_ids = get_input_ids(
                pair_ids) if pair_ids is not None else None
            input_ids.append((first_ids, second_ids))

        if stride > 0 and second_ids is not None:
            kwargs['batch_text_or_text_pairs'] = batch_text_or_text_pairs
        else:
            if return_offsets_mapping:
                has_pair = False
                if len(batch_text_or_text_pairs) > 0:
                    if isinstance(batch_text_or_text_pairs[0], (list, tuple)):
                        has_pair = True
                kwargs['texts'] = None
                kwargs['text_pairs'] = None
                if has_pair:
                    kwargs['texts'] = [
                        text[0] for text in batch_text_or_text_pairs
                    ]
                    kwargs['text_pairs'] = [
                        text[1] for text in batch_text_or_text_pairs
                    ]
                else:
                    kwargs['texts'] = [
                        text for text in batch_text_or_text_pairs
                    ]

        batch_outputs = self._batch_prepare_for_model(
            input_ids,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_position_ids=return_position_ids,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_dict=return_dict,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            return_tensors=return_tensors,
            verbose=verbose,
            **kwargs)

        return batch_outputs

    def _batch_prepare_for_model(
            self,
            batch_ids_pairs: List[Union[PreTokenizedInputPair, Tuple[List[int],
                                                                     None]]],
            add_special_tokens: bool = True,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            truncation_strategy: TruncationStrategy = TruncationStrategy.
        DO_NOT_TRUNCATE,
            max_length: Optional[int] = None,
            stride: int = 0,
            pad_to_multiple_of: Optional[int] = None,
            return_position_ids: Optional[bool] = None,
            return_tensors: Optional[str] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_dict: bool = True,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            **kwargs):
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens

        Args:
            batch_ids_pairs: list of tokenized input ids or input ids pairs
        """
        if return_token_type_ids and not add_special_tokens:
            raise ValueError(
                "Asking to return token_type_ids while setting add_special_tokens to False "
                "results in an undefined behavior. Please set add_special_tokens to True or "
                "set return_token_type_ids to None.")

        batch_outputs = {}
        batch_outputs_list = []
        for example_id, (first_ids, second_ids) in enumerate(batch_ids_pairs):
            if stride > 0 and second_ids is not None:
                if return_token_type_ids is None:
                    return_token_type_ids = "token_type_ids" in self.model_input_names
                if return_attention_mask is None:
                    return_attention_mask = "attention_mask" in self.model_input_names

                max_len_for_pair = max_length - len(first_ids) - (
                    self.num_special_tokens_to_add(
                        pair=True) if add_special_tokens else 0)

                text, text_pair = kwargs['batch_text_or_text_pairs'][example_id]
                token_offset_mapping = self.get_offset_mapping(text)
                token_pair_offset_mapping = self.get_offset_mapping(text_pair)

                offset = 0
                while offset < len(second_ids):
                    encoded_inputs = {}
                    length = len(second_ids) - offset
                    if length > max_len_for_pair:
                        length = max_len_for_pair

                    ids = first_ids
                    pair_ids = second_ids[offset:offset + length]
                    pair = bool(pair_ids is not None)
                    mapping = token_offset_mapping
                    pair_mapping = token_pair_offset_mapping[offset:offset +
                                                             length]
                    if add_special_tokens:
                        offset_mapping = self.build_offset_mapping_with_special_tokens(
                            mapping, pair_mapping)
                        sequence = self.build_inputs_with_special_tokens(
                            ids, pair_ids)
                        token_type_ids = self.create_token_type_ids_from_sequences(
                            ids, pair_ids)
                    else:
                        offset_mapping = mapping + pair_mapping
                        sequence = ids + pair_ids if pair else ids
                        token_type_ids = [0] * len(ids) + ([0] * len(pair_ids)
                                                           if pair else [])
                    encoded_inputs['offset_mapping'] = offset_mapping
                    # Build output dictionnary
                    encoded_inputs["input_ids"] = sequence
                    if return_token_type_ids:
                        encoded_inputs["token_type_ids"] = token_type_ids
                    if return_special_tokens_mask:
                        if add_special_tokens:
                            encoded_inputs[
                                "special_tokens_mask"] = self.get_special_tokens_mask(
                                    ids, pair_ids)
                        else:
                            encoded_inputs["special_tokens_mask"] = [
                                0
                            ] * len(sequence)

                    # Check lengths
                    self._eventual_warn_about_too_long_sequence(
                        encoded_inputs["input_ids"], max_length, verbose)
                    if return_position_ids:
                        encoded_inputs["position_ids"] = list(
                            range(len(encoded_inputs["input_ids"])))

                    if return_length:
                        encoded_inputs["length"] = len(
                            encoded_inputs["input_ids"])
                        encoded_inputs["seq_len"] = encoded_inputs["length"]

                    encoded_inputs['overflow_to_sample'] = example_id

                    for key, value in encoded_inputs.items():
                        if key not in batch_outputs:
                            batch_outputs[key] = []
                        batch_outputs[key].append(value)

                    if offset + length == len(second_ids):
                        break
                    offset += min(length, stride)
            else:
                if return_offsets_mapping:
                    kwargs['text'] = kwargs['texts'][example_id]
                    kwargs['text_pair'] = None
                    if kwargs['text_pairs'] is not None:
                        kwargs['text_pair'] = kwargs['text_pairs'][example_id]

                encoded_inputs = self.prepare_for_model(
                    first_ids,
                    second_ids,
                    add_special_tokens=add_special_tokens,
                    padding=PaddingStrategy.DO_NOT_PAD.
                    value,  # we pad in batch afterward
                    truncation=truncation_strategy.value,
                    max_length=max_length,
                    stride=stride,
                    pad_to_multiple_of=None,  # we pad in batch afterward
                    return_position_ids=
                    return_position_ids,  # we pad in batch afterward
                    return_attention_mask=False,  # we pad in batch afterward
                    return_token_type_ids=return_token_type_ids,
                    return_overflowing_tokens=return_overflowing_tokens,
                    return_special_tokens_mask=return_special_tokens_mask,
                    return_offsets_mapping=return_offsets_mapping,
                    return_length=return_length,
                    return_tensors=
                    None,  # We convert the whole batch to tensors at the end
                    prepend_batch_axis=False,
                    verbose=verbose,
                    **kwargs)
                for key, value in encoded_inputs.items():
                    if key not in batch_outputs:
                        batch_outputs[key] = []
                    batch_outputs[key].append(value)

        batch_outputs = self.pad(
            batch_outputs,
            padding=padding_strategy.value,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )
        if return_dict:
            batch_outputs = BatchEncoding(batch_outputs,
                                          tensor_type=return_tensors)
            return batch_outputs
        else:
            for k, v in batch_outputs.items():
                for i in range(len(v)):
                    if i >= len(batch_outputs_list):
                        batch_outputs_list.append({k: v[i]})
                    else:
                        batch_outputs_list[i][k] = v[i]
            return batch_outputs_list


    def get_offset_mapping(self, text):
        """
        Returns the map of tokens and the start and end index of their start and end character.
        Modified from https://github.com/bojone/bert4keras/blob/master/bert4keras/tokenizers.py#L372    
        """
        if text is None:
            return None
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(
                    sub_token if sub_token != self.unk_token else token)

        normalized_text, char_mapping = '', []

        for i, ch in enumerate(text):
            if hasattr(self, "do_lower_case") and self.do_lower_case:
                ch = ch.lower()
            ch = unicodedata.normalize('NFD', ch)
            ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])

            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or _is_control(c))
            ])
            normalized_text += ch

            char_mapping.extend([i] * len(ch))
        text, token_mapping, offset = normalized_text, [], 0

        for token in split_tokens:

            if token[:2] == '##':
                token = token[2:]

            start = text[offset:].index(token) + offset

            end = start + len(token)

            token_mapping.append(
                (char_mapping[start], char_mapping[end - 1] + 1))
            offset = end

        return token_mapping

    def convert_tokens_to_ids(self, tokens):
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))

        return ids
    
    def _convert_token_to_id_with_added_voc(self, token):
        if token is None:
            return None

        if token in self.added_tokens_encoder:
            return self.added_tokens_encoder[token]
        return self._convert_token_to_id(token)

    def _convert_token_to_id(self, token):

        return self.vocab.to_indices(token)

    def prepare_for_tokenization(self,
                                text,
                                is_split_into_words=False,
                                **kwargs):
        """
        Performs any necessary transformations before tokenization.
        """

        return (text, kwargs)

    def tokenize(self, text: TextInput, **kwargs) -> List[str]:
        """
        Converts a string in a sequence of tokens, using the tokenizer.

        Split in words for word-based vocabulary or sub-words for sub-word-based vocabularies
        (BPE/SentencePieces/WordPieces). Takes care of added tokens.

        Args:
            text (`str`):
                The sequence to be encoded.
            **kwargs (additional keyword arguments):
                Passed along to the model-specific `prepare_for_tokenization` preprocessing method.

        Returns:
            `List[str]`: The list of tokens.
        """
        # Simple mapping string => AddedToken for special tokens with specific tokenization behaviors
        all_special_tokens_extended = dict(
            (str(t), t) for t in self.all_special_tokens_extended
            if isinstance(t, AddedToken))

        text, kwargs = self.prepare_for_tokenization(text, **kwargs)

        # TODO: should this be in the base class?
        if hasattr(self, "do_lower_case") and self.do_lower_case:
            # convert non-special tokens to lowercase
            escaped_special_toks = [
                re.escape(s_tok) for s_tok in (self.unique_no_split_tokens +
                                               self.all_special_tokens)
            ]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            text = re.sub(pattern,
                          lambda m: m.groups()[0] or m.groups()[1].lower(),
                          text)

        no_split_token = set(self.unique_no_split_tokens)
        tokens = self.tokens_trie.split(text)
        # ["This is something", "<special_token_1>", "  else"]
        for i, token in enumerate(tokens):
            if token in no_split_token:
                tok_extended = all_special_tokens_extended.get(token, None)
                left = tokens[i - 1] if i > 0 else None
                right = tokens[i + 1] if i < len(tokens) - 1 else None
                if isinstance(tok_extended, AddedToken):
                    if tok_extended.rstrip and right:
                        # A bit counter-intuitive but we strip the left of the string
                        # since tok_extended.rstrip means the special token is eating all white spaces on its right
                        tokens[i + 1] = right.lstrip()
                    # Strip white spaces on the left
                    if tok_extended.lstrip and left:
                        tokens[i - 1] = left.rstrip()  # Opposite here
                else:
                    # We strip left and right by default
                    if right:
                        tokens[i + 1] = right.lstrip()
                    if left:
                        tokens[i - 1] = left.rstrip()
        # ["This is something", "<special_token_1>", "else"]
        tokenized_text = []
        for token in tokens:
            # Need to skip eventual empty (fully stripped) tokens
            if not token:
                continue
            if token in no_split_token:
                tokenized_text.append(token)
            else:
                tokenized_text.extend(self._tokenize(token))
        # ["This", " is", " something", "<special_token_1>", "else"]
        return tokenized_text
    
    def _tokenize(self, text):
        r"""
        End-to-end tokenization for ERNIE models.

        Args:
            text (str): The text to be tokenized.
        
        Returns:
            List[str]: A list of string representing converted tokens.
        """
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens
    
    def get_special_tokens_mask(self,
                                token_ids_0,
                                token_ids_1=None,
                                already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``encode`` methods.
        Args:
            token_ids_0 (List[int]): List of ids of the first sequence.
            token_ids_1 (List[int], optional): List of ids of the second sequence.
            already_has_special_tokens (bool, optional): Whether or not the token list is already
                formatted with special tokens for the model. Defaults to None.
        Returns:
            results (List[int]): The list of integers in the range [0, 1]:
                1 for a special token, 0 for a sequence token.
        """
        return [0] * (
            (len(token_ids_1) if token_ids_1 else 0) + len(token_ids_0))
    
    def create_token_type_ids_from_sequences(self,
                                             token_ids_0,
                                             token_ids_1=None):
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task.
        Should be overridden in a subclass if the model has a special way of building those.
        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).
        Args:
            token_ids_0 (List[int]):
                List of IDs.
            token_ids_1 (List[int], optional):
                Optional second list of IDs for sequence pairs.
        Returns:
            List[int]: List of token_type_id according to the given sequence(s).
        """
        _sep = [self.sep_token_id]
        _cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(_cls + token_ids_0 + _sep) * [0]
        return len(_cls + token_ids_0 + _sep) * [0] + len(token_ids_1 +
                                                          _sep) * [1]
    
    def _eventual_warn_about_too_long_sequence(self, ids: List[int],
                                               max_length: Optional[int],
                                               verbose: bool):
        """
        Depending on the input and internal state we might trigger a warning about a sequence that is too long for its
        corresponding model

        Args:
            ids (`List[str]`): The ids produced by the tokenization
            max_length (`int`, *optional*): The max_length desired (does not trigger a warning if it is set)
            verbose (`bool`): Whether or not to print more information and warnings.

        """
        if max_length is None and len(ids) > self.model_max_length and verbose:
            if not self.deprecation_warnings.get(
                    "sequence-length-is-longer-than-the-specified-maximum",
                    False):
                warnings.warn(
                    "Token indices sequence length is longer than the specified maximum sequence length "
                    f"for this model ({len(ids)} > {self.model_max_length}). Running this sequence through the model "
                    "will result in indexing errors")
            self.deprecation_warnings[
                "sequence-length-is-longer-than-the-specified-maximum"] = True
    
    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            encoded_inputs:
                Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in self.padding_side:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                >= 7.5 (Volta).
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names or "attention_mask" in encoded_inputs

        required_input = encoded_inputs[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (
                max_length % pad_to_multiple_of != 0):
            max_length = (
                (max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(
            required_input) != max_length

        # Initialize attention mask if not present.
        if return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * len(required_input)

        if needs_to_be_padded:
            difference = max_length - len(required_input)

            if self.padding_side == "right":
                if return_attention_mask:

                    encoded_inputs["attention_mask"] = encoded_inputs[
                        "attention_mask"] + [0] * difference
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] +
                        [self.pad_token_type_id] * difference)
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs[
                        "special_tokens_mask"] + [1] * difference
                if "offset_mapping" in encoded_inputs:
                    encoded_inputs["offset_mapping"] = encoded_inputs[
                        "offset_mapping"] + [(0, 0)] * difference
                if "position_ids" in encoded_inputs:
                    encoded_inputs["position_ids"] = encoded_inputs[
                        "position_ids"] + [0] * difference
                encoded_inputs[self.model_input_names[
                    0]] = required_input + [self.pad_token_id] * difference
            elif self.padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [
                        0
                    ] * difference + encoded_inputs["attention_mask"]
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = [
                        self.pad_token_type_id
                    ] * difference + encoded_inputs["token_type_ids"]
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [
                        1
                    ] * difference + encoded_inputs["special_tokens_mask"]
                if "offset_mapping" in encoded_inputs:
                    encoded_inputs["offset_mapping"] = [
                        (0, 0)
                    ] * difference + encoded_inputs["offset_mapping"]
                if "position_ids" in encoded_inputs:
                    encoded_inputs["position_ids"] = [
                        0
                    ] * difference + encoded_inputs["position_ids"]
                encoded_inputs[self.model_input_names[
                    0]] = [self.pad_token_id] * difference + required_input
            else:
                raise ValueError("Invalid padding strategy:" +
                                 str(self.padding_side))

        return encoded_inputs
    
    def pad(self, encoded_inputs: Union[BatchEncoding, List[BatchEncoding],
                              Dict[str, EncodedInput], Dict[str,
                                                            List[EncodedInput]],
                              List[Dict[str, EncodedInput]], ],
            padding: Union[bool, str, PaddingStrategy] = True,
            max_length: Optional[int] = None,
            pad_to_multiple_of: Optional[int] = None,
            return_attention_mask: Optional[bool] = None,
            return_tensors: Optional[str] = None,
            verbose: bool = True,
    ) -> BatchEncoding:
        """
        """
        # If we have a list of dicts, let's convert it in a dict of lists
        if isinstance(encoded_inputs,
                      (list, tuple)) and isinstance(encoded_inputs[0],
                                                    (dict, BatchEncoding)):
            encoded_inputs = {
                key: [example[key] for example in encoded_inputs]
                for key in encoded_inputs[0].keys()
            }

        # The model's main input name, usually `input_ids`, has be passed for padding
        if self.model_input_names[0] not in encoded_inputs:
            raise ValueError(
                "You should supply an encoding or a list of encodings to this method "
                f"that includes {self.model_input_names[0]}, but you provided {list(encoded_inputs.keys())}"
            )

        required_input = encoded_inputs[self.model_input_names[0]]

        if not required_input:
            if return_attention_mask:
                encoded_inputs["attention_mask"] = []
            return encoded_inputs

        # If we have Paddle/NumPy tensors/arrays as inputs, we cast them as python objects
        # and rebuild them afterwards if no return_tensors is specified

        first_element = required_input[0]
        if isinstance(first_element, (list, tuple)):
            # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
            for item in required_input:
                if len(item) != 0:
                    first_element = item[0]
                    break
        # At this state, if `first_element` is still a list/tuple, it's an empty one so there is nothing to do.
        if not isinstance(first_element, (int, list, tuple)):
            for key, value in encoded_inputs.items():
                encoded_inputs[key] = to_py_obj(value)

        # Convert padding_strategy in PaddingStrategy
        padding_strategy, _, max_length, _ = self._get_padding_truncation_strategies(
            padding=padding, max_length=max_length, verbose=verbose)

        required_input = encoded_inputs[self.model_input_names[0]]
        if required_input and not isinstance(required_input[0], (list, tuple)):
            encoded_inputs = self._pad(
                encoded_inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask)
            return BatchEncoding(encoded_inputs, tensor_type=return_tensors)

        batch_size = len(required_input)
        assert all(
            len(v) == batch_size for v in encoded_inputs.values()
        ), "Some items in the output dictionary have a different batch size than others."

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = max(len(inputs) for inputs in required_input)
            padding_strategy = PaddingStrategy.MAX_LENGTH

        batch_outputs = {}
        for i in range(batch_size):
            inputs = dict((k, v[i]) for k, v in encoded_inputs.items())
            outputs = self._pad(
                inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        return BatchEncoding(batch_outputs, tensor_type=return_tensors)


    
    def prepare_for_model(self,
                        ids,
                        pair_ids=None,
                        padding: Union[bool, str, PaddingStrategy] = False,
                        truncation: Union[bool, str,
                                        TruncationStrategy] = False,
                        max_length: Optional[int] = None,
                        stride: int = 0,
                        pad_to_multiple_of: Optional[int] = None,
                        return_tensors: Optional[Union[str,
                                                        TensorType]] = None,
                        return_position_ids=False,
                        return_token_type_ids: Optional[bool] = None,
                        return_attention_mask: Optional[bool] = None,
                        return_length=False,
                        return_overflowing_tokens=False,
                        return_special_tokens_mask=False,
                        return_offsets_mapping=False,
                        add_special_tokens=True,
                        verbose: bool = True,
                        prepend_batch_axis: bool = False,
                        **kwargs):
        """
        Performs tokenization and uses the tokenized tokens to prepare model
        inputs. It supports sequence or sequence pair as input, and batch input
        is not allowed.
        """
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        if return_token_type_ids and not add_special_tokens:
            raise ValueError(
                "Asking to return token_type_ids while setting add_special_tokens to False "
                "results in an undefined behavior. Please set add_special_tokens to True or "
                "set return_token_type_ids to None.")

        if (return_overflowing_tokens
                and truncation_strategy == TruncationStrategy.LONGEST_FIRST
                and pair_ids is not None):
            raise ValueError(
                "Not possible to return overflowing tokens for pair of sequences with the "
                "`longest_first`. Please select another truncation strategy than `longest_first`, "
                "for instance `only_second` or `only_first`.")

        # Load from model defaults
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        encoded_inputs = {}
        # Truncation: Handle max sequence length
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(
            pair=pair) if add_special_tokens else 0)

        overflowing_tokens = []

        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and total_len > max_length:
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation_strategy,
                stride=stride,
            )
        if return_overflowing_tokens:
            encoded_inputs["overflowing_tokens"] = overflowing_tokens
            encoded_inputs["num_truncated_tokens"] = total_len - max_length

        # Add special tokens
        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
            token_type_ids = self.create_token_type_ids_from_sequences(
                ids, pair_ids)
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([0] *
                                               len(pair_ids) if pair else [])

        # Build output dictionnary
        encoded_inputs["input_ids"] = sequence
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs[
                    "special_tokens_mask"] = self.get_special_tokens_mask(
                        ids, pair_ids)
            else:
                encoded_inputs["special_tokens_mask"] = [0] * len(sequence)

        if return_offsets_mapping and 'text' in kwargs and 'text_pair' in kwargs:
            text = kwargs.pop('text')
            text_pair = kwargs.pop('text_pair')

            token_offset_mapping = self.get_offset_mapping(text)
            token_pair_offset_mapping = self.get_offset_mapping(text_pair)
            if max_length and total_len > max_length:
                token_offset_mapping, token_pair_offset_mapping, _ = self.truncate_sequences(
                    token_offset_mapping,
                    pair_ids=token_pair_offset_mapping,
                    num_tokens_to_remove=total_len - max_length,
                    truncation_strategy=truncation_strategy,
                    stride=stride)
            if add_special_tokens:
                offset_mapping = self.build_offset_mapping_with_special_tokens(
                    token_offset_mapping, token_pair_offset_mapping)
            else:
                offset_mapping = token_offset_mapping + token_pair_offset_mapping if token_pair_offset_mapping else token_offset_mapping
            encoded_inputs['offset_mapping'] = offset_mapping

        # Check lengths
        self._eventual_warn_about_too_long_sequence(encoded_inputs["input_ids"],
                                                    max_length, verbose)

        if return_position_ids:
            encoded_inputs["position_ids"] = list(
                range(len(encoded_inputs["input_ids"])))

        if padding_strategy != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            encoded_inputs = self.pad(
                encoded_inputs,
                max_length=max_length,
                padding=padding_strategy.value,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

        if return_length:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])
            #for compatibility
            encoded_inputs["seq_len"] = encoded_inputs["length"]

        batch_outputs = BatchEncoding(encoded_inputs,
                                      tensor_type=return_tensors,
                                      prepend_batch_axis=prepend_batch_axis)

        return batch_outputs

    
    def truncate_sequences(self,
                           ids,
                           pair_ids=None,
                           num_tokens_to_remove=0,
                           truncation_strategy='longest_first',
                           stride=0):
        """
        Truncates a sequence pair in place to the maximum length.
        """
        if num_tokens_to_remove <= 0:
            return ids, pair_ids, []

        if truncation_strategy == 'longest_first':
            overflowing_tokens = []
            for _ in range(num_tokens_to_remove):
                if pair_ids is None or len(ids) > len(pair_ids):
                    overflowing_tokens = [ids[-1]] + overflowing_tokens
                    ids = ids[:-1]
                else:
                    pair_ids = pair_ids[:-1]
            window_len = min(len(ids), stride)
            if window_len > 0:
                overflowing_tokens = ids[-window_len:] + overflowing_tokens
        elif truncation_strategy == 'only_first':
            assert len(ids) > num_tokens_to_remove
            window_len = min(len(ids), stride + num_tokens_to_remove)
            overflowing_tokens = ids[-window_len:]
            ids = ids[:-num_tokens_to_remove]
        elif truncation_strategy == 'only_second':
            assert pair_ids is not None and len(pair_ids) > num_tokens_to_remove
            window_len = min(len(pair_ids), stride + num_tokens_to_remove)
            overflowing_tokens = pair_ids[-window_len:]
            pair_ids = pair_ids[:-num_tokens_to_remove]
        elif truncation_strategy == 'do_not_truncate':
            raise ValueError(
                "Input sequence are too long for max_length. Please select a truncation strategy."
            )
        else:
            raise ValueError(
                "Truncation_strategy should be selected in ['longest_first', 'only_first', 'only_second', 'do_not_truncate']"
            )
        return (ids, pair_ids, overflowing_tokens)
    
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        _cls = [self.cls_token_id]
        _sep = [self.sep_token_id]
        return _cls + token_ids_0 + _sep + token_ids_1 + _sep
    
    def build_offset_mapping_with_special_tokens(self,
                                                 offset_mapping_0,
                                                 offset_mapping_1=None):
        """
        Build offset map from a pair of offset map by concatenating and adding offsets of special tokens.

        Should be overridden in a subclass if the model has a special way of building those.
        """
        if offset_mapping_1 is None:
            return [(0, 0)] + offset_mapping_0 + [(0, 0)]

        return [(0, 0)] + offset_mapping_0 + [(0, 0)
                                              ] + offset_mapping_1 + [(0, 0)]

    def num_special_tokens_to_add(self, pair):
        """
        Returns the number of added tokens when encoding a sequence with special tokens.
        """
        token_ids_0 = []
        token_ids_1 = []
        return len(
            self.build_inputs_with_special_tokens(
                token_ids_0, token_ids_1 if pair else None))


# import paddle2onnx
# import onnxruntime as ort  

# class InferBackend(object):

#     def __init__(self,
#                  model_path_prefix,
#                  device='cpu',
#                  use_quantize=False,
#                  use_fp16=False):
#         print(">>> [InferBackend] Creating Engine ...")
#         onnx_model = paddle2onnx.command.c_paddle_to_onnx(
#             model_file=model_path_prefix + ".pdmodel",
#             params_file=model_path_prefix + ".pdiparams",
#             opset_version=13,
#             enable_onnx_checker=True)
#         infer_model_dir = model_path_prefix.rsplit("/", 1)[0]
#         float_onnx_file = os.path.join(infer_model_dir, "model.onnx")
#         with open(float_onnx_file, "wb") as f:
#             f.write(onnx_model)

#         if device == "gpu":
#             providers = ['CUDAExecutionProvider']
#             print(">>> [InferBackend] Use GPU to inference ...")
#             if use_fp16:
#                 print(">>> [InferBackend] Use FP16 to inference ...")
#                 from onnxconverter_common import float16
#                 import onnx
#                 fp16_model_file = os.path.join(infer_model_dir,
#                                                "fp16_model.onnx")
#                 onnx_model = onnx.load_model(float_onnx_file)
#                 trans_model = float16.convert_float_to_float16(
#                     onnx_model, keep_io_types=True)
#                 onnx.save_model(trans_model, fp16_model_file)
#                 onnx_model = fp16_model_file
#         else:
#             providers = ['CPUExecutionProvider']
#             print(">>> [InferBackend] Use CPU to inference ...")

#         sess_options = ort.SessionOptions()
#         self.predictor = ort.InferenceSession(onnx_model,
#                                               sess_options=sess_options,
#                                               providers=providers)
#         if device == "gpu":
#             assert 'CUDAExecutionProvider' in self.predictor.get_providers(), \
#                 f"The environment for GPU inference is not set properly. " \
#                 "A possible cause is that you had installed both onnxruntime and onnxruntime-gpu. " \
#                 "Please run the following commands to reinstall: \n " \
#                 "1) pip uninstall -y onnxruntime onnxruntime-gpu \n 2) pip install onnxruntime-gpu"
#         print(">>> [InferBackend] Engine Created ...")

#     def infer(self, input_dict: dict):
#         result = self.predictor.run(None, input_dict)
#         return result


class UIEPredictor(object):

    def __init__(self, max_seq_len, batch_size, schema, position_prob=0.5):

        self._tokenizer = MyTokenizer('/root/lijiaqi/UIE/vocab.txt')
        # AutoTokenizer.from_pretrained("ernie-3.0-base-zh",
        #                                                 use_faster=True)
        self._position_prob = position_prob
        self._max_seq_len = max_seq_len
        self._batch_size = batch_size
        self._schema_tree = None
        self.set_schema(schema)

        # self.inference_backend = InferBackend('/home/user/lijiaqi/PaddleNLP/model_zoo/uie/export/inference', device='cpu', use_fp16=False)

    def set_schema(self, schema):
        if isinstance(schema, dict) or isinstance(schema, str):
            schema = [schema]
        self._schema_tree = self._build_tree(schema)

    @classmethod
    def _build_tree(cls, schema, name='root'):
        """
        Build the schema tree.
        """
        schema_tree = SchemaTree(name)
        for s in schema:
            if isinstance(s, str):
                schema_tree.add_child(SchemaTree(s))
            elif isinstance(s, dict):
                for k, v in s.items():
                    if isinstance(v, str):
                        child = [v]
                    elif isinstance(v, list):
                        child = v
                    else:
                        raise TypeError(
                            "Invalid schema, value for each key:value pairs should be list or string"
                            "but {} received".format(type(v)))
                    schema_tree.add_child(cls._build_tree(child, name=k))
            else:
                raise TypeError(
                    "Invalid schema, element should be string or dict, "
                    "but {} received".format(type(s)))
        return schema_tree

    def _single_stage_predict(self, inputs):
        input_texts = []
        prompts = []
        for i in range(len(inputs)):
            input_texts.append(inputs[i]["text"])
            prompts.append(inputs[i]["prompt"])
        # max predict length should exclude the length of prompt and summary tokens
        max_predict_len = self._max_seq_len - len(max(prompts)) - 3
        short_input_texts, self.input_mapping = self._auto_splitter(
            input_texts, max_predict_len, split_sentence=False)

        short_texts_prompts = []
        for k, v in self.input_mapping.items():
            short_texts_prompts.extend([prompts[k] for i in range(len(v))])
        short_inputs = [{
            "text": short_input_texts[i],
            "prompt": short_texts_prompts[i]
        } for i in range(len(short_input_texts))]

        prompts = []
        texts = []
        for s in short_inputs:
            prompts.append(s['prompt'])
            texts.append(s['text'])
        encoded_inputs = self._tokenizer(text=prompts,
                                         text_pair=texts,
                                         truncation=True,
                                         max_seq_len=self._max_seq_len,
                                         pad_to_max_seq_len=True,
                                         return_attention_mask=True,
                                         return_position_ids=True,
                                         return_tensors='np',
                                         return_offsets_mapping=True)
        offset_maps = encoded_inputs["offset_mapping"]

        start_probs = []
        end_probs = []
        for idx in range(0, len(texts), self._batch_size):
            l, r = idx, idx + self._batch_size
            input_dict = {
                "input_ids": encoded_inputs['input_ids'][l:r].astype('int64'),
                "token_type_ids": encoded_inputs['token_type_ids'][l:r].astype('int64'),
                "pos_ids": encoded_inputs['position_ids'][l:r].astype('int64'),
                "att_mask": encoded_inputs["attention_mask"][l:r].astype('int64')
            }


            context = allocate_res(0)
            model_id = load_model(MODEL_PATH) 
            # print("model_id:{}".format(model_id))
            input_num, output_num = get_model_data(model_id)
            malloc_device(input_num, output_num)

            _data_interaction_in(input_dict["input_ids"], input_dict["token_type_ids"], input_dict["pos_ids"], input_dict["att_mask"])
            _gen_dataset("in")
            _gen_dataset("out")
            inference(model_id, load_input_dataset, load_output_dataset)
            _destroy_data_set_buffer()
            res = []
            _data_interaction_out(res)
            starts, ends = print_result(res)
            release(model_id, context)


            start_probs.extend(starts)
            end_probs.extend(ends)

        start_ids_list = get_bool_ids_greater_than(start_probs,
                                                   limit=self._position_prob,
                                                   return_prob=True)
        end_ids_list = get_bool_ids_greater_than(end_probs,
                                                 limit=self._position_prob,
                                                 return_prob=True)

        input_ids = input_dict['input_ids']
        sentence_ids = []
        probs = []
        for start_ids, end_ids, ids, offset_map in zip(start_ids_list,
                                                       end_ids_list,
                                                       input_ids.tolist(),
                                                       offset_maps.tolist()):
            for i in reversed(range(len(ids))):
                if ids[i] != 0:
                    ids = ids[:i]
                    break
            span_list = get_span(start_ids, end_ids, with_prob=True)
            sentence_id, prob = get_id_and_prob(span_list, offset_map)
            sentence_ids.append(sentence_id)
            probs.append(prob)

        results = self._convert_ids_to_results(short_inputs, sentence_ids,
                                               probs)
        results = self._auto_joiner(results, short_input_texts,
                                    self.input_mapping)
        
        # print(results)
        # exit()
        return results

    def _auto_splitter(self, input_texts, max_text_len, split_sentence=False):
        '''
        Split the raw texts automatically for model inference.
        '''
        input_mapping = {}
        short_input_texts = []
        cnt_org = 0
        cnt_short = 0
        for text in input_texts:
            if not split_sentence:
                sens = [text]
            else:
                sens = cut_chinese_sent(text)
            for sen in sens:
                lens = len(sen)
                if lens <= max_text_len:
                    short_input_texts.append(sen)
                    if cnt_org not in input_mapping.keys():
                        input_mapping[cnt_org] = [cnt_short]
                    else:
                        input_mapping[cnt_org].append(cnt_short)
                    cnt_short += 1
                else:
                    temp_text_list = [
                        sen[i:i + max_text_len]
                        for i in range(0, lens, max_text_len)
                    ]
                    short_input_texts.extend(temp_text_list)
                    short_idx = cnt_short
                    cnt_short += math.ceil(lens / max_text_len)
                    temp_text_id = [
                        short_idx + i for i in range(cnt_short - short_idx)
                    ]
                    if cnt_org not in input_mapping.keys():
                        input_mapping[cnt_org] = temp_text_id
                    else:
                        input_mapping[cnt_org].extend(temp_text_id)
            cnt_org += 1
        return short_input_texts, input_mapping

    def _auto_joiner(self, short_results, short_inputs, input_mapping):
        concat_results = []
        is_cls_task = False
        for short_result in short_results:
            if short_result == []:
                continue
            elif 'start' not in short_result[0].keys(
            ) and 'end' not in short_result[0].keys():
                is_cls_task = True
                break
            else:
                break
        for k, vs in input_mapping.items():
            if is_cls_task:
                cls_options = {}
                single_results = []
                for v in vs:
                    if len(short_results[v]) == 0:
                        continue
                    if short_results[v][0]['text'] not in cls_options.keys():
                        cls_options[short_results[v][0]['text']] = [
                            1, short_results[v][0]['probability']
                        ]
                    else:
                        cls_options[short_results[v][0]['text']][0] += 1
                        cls_options[short_results[v][0]['text']][
                            1] += short_results[v][0]['probability']
                if len(cls_options) != 0:
                    cls_res, cls_info = max(cls_options.items(),
                                            key=lambda x: x[1])
                    concat_results.append([{
                        'text':
                        cls_res,
                        'probability':
                        cls_info[1] / cls_info[0]
                    }])
                else:
                    concat_results.append([])
            else:
                offset = 0
                single_results = []
                for v in vs:
                    if v == 0:
                        single_results = short_results[v]
                        offset += len(short_inputs[v])
                    else:
                        for i in range(len(short_results[v])):
                            if 'start' not in short_results[v][
                                    i] or 'end' not in short_results[v][i]:
                                continue
                            short_results[v][i]['start'] += offset
                            short_results[v][i]['end'] += offset
                        offset += len(short_inputs[v])
                        single_results.extend(short_results[v])
                concat_results.append(single_results)
        return concat_results

    def _convert_ids_to_results(self, examples, sentence_ids, probs):
        """
        Convert ids to raw text in a single stage.
        """
        results = []
        for example, sentence_id, prob in zip(examples, sentence_ids, probs):
            if len(sentence_id) == 0:
                results.append([])
                continue
            result_list = []
            text = example["text"]
            prompt = example["prompt"]
            for i in range(len(sentence_id)):
                start, end = sentence_id[i]
                if start < 0 and end >= 0:
                    continue
                if end < 0:
                    start += (len(prompt) + 1)
                    end += (len(prompt) + 1)
                    result = {"text": prompt[start:end], "probability": prob[i]}
                    result_list.append(result)
                else:
                    result = {
                        "text": text[start:end],
                        "start": start,
                        "end": end,
                        "probability": prob[i]
                    }
                    result_list.append(result)
            results.append(result_list)
        return results

    def _multi_stage_predict(self, data):
        """
        Traversal the schema tree and do multi-stage prediction.
        """
        results = [{} for _ in range(len(data))]
        # input check to early return
        if len(data) < 1 or self._schema_tree is None:
            return results

        # copy to stay `self._schema_tree` unchanged
        schema_list = self._schema_tree.children[:]
        while len(schema_list) > 0:
            node = schema_list.pop(0)
            examples = []
            input_map = {}
            cnt = 0
            idx = 0
            if not node.prefix:
                for one_data in data:
                    examples.append({
                        "text": one_data,
                        "prompt": dbc2sbc(node.name)
                    })
                    input_map[cnt] = [idx]
                    idx += 1
                    cnt += 1
            else:
                for pre, one_data in zip(node.prefix, data):
                    if len(pre) == 0:
                        input_map[cnt] = []
                    else:
                        for p in pre:
                            examples.append({
                                "text": one_data,
                                "prompt": dbc2sbc(p + node.name)
                            })
                        input_map[cnt] = [i + idx for i in range(len(pre))]
                        idx += len(pre)
                    cnt += 1
            if len(examples) == 0:
                result_list = {}
            else:
                result_list = self._single_stage_predict(examples)
            
        
        # return result_dict
            if not node.parent_relations:
                relations = [[] for i in range(len(data))]
                for k, v in input_map.items():
                    for idx in v:
                        if len(result_list[idx]) == 0:
                            continue
                        if node.name not in results[k].keys():
                            results[k][node.name] = result_list[idx]
                        else:
                            results[k][node.name].extend(result_list[idx])
                    if node.name in results[k].keys():
                        relations[k].extend(results[k][node.name])
            else:
                relations = node.parent_relations
                for k, v in input_map.items():
                    for i in range(len(v)):
                        if len(result_list[v[i]]) == 0:
                            continue
                        if "relations" not in relations[k][i].keys():
                            relations[k][i]["relations"] = {
                                node.name: result_list[v[i]]
                            }
                        elif node.name not in relations[k][i]["relations"].keys(
                        ):
                            relations[k][i]["relations"][
                                node.name] = result_list[v[i]]
                        else:
                            relations[k][i]["relations"][node.name].extend(
                                result_list[v[i]])
                new_relations = [[] for i in range(len(data))]
                for i in range(len(relations)):
                    for j in range(len(relations[i])):
                        if "relations" in relations[i][j].keys(
                        ) and node.name in relations[i][j]["relations"].keys():
                            for k in range(
                                    len(relations[i][j]["relations"][
                                        node.name])):
                                new_relations[i].append(
                                    relations[i][j]["relations"][node.name][k])
                relations = new_relations

            prefix = [[] for _ in range(len(data))]
            for k, v in input_map.items():
                for idx in v:
                    for i in range(len(result_list[idx])):
                        prefix[k].append(result_list[idx][i]["text"] + "的")

            for child in node.children:
                child.prefix = prefix
                child.parent_relations = relations
                schema_list.append(child)

        return results
 
    def _infer(self, data):
        return self.inference_backend.infer(data)

    def predict(self, input_data):
        results = self._multi_stage_predict(input_data)
        return results


class SchemaTree(object):
    """
    Implementataion of SchemaTree
    """
    def __init__(self, name='root', children=None):
        self.name = name
        self.children = []
        self.prefix = None
        self.parent_relations = None
        if children is not None:
            for child in children:
                self.add_child(child)

    def __repr__(self):
        return self.name

    def add_child(self, node):
        assert isinstance(
            node, SchemaTree
        ), "The children of a node should be an instacne of SchemaTree."
        self.children.append(node)


def get_bool_ids_greater_than(probs, limit=0.5, return_prob=False):
    """
    Get idx of the last dimension in probability arrays, which is greater than a limitation.
    """
    probs = np.array(probs)
    dim_len = len(probs.shape)
    if dim_len > 1:
        result = []
        for p in probs:
            result.append(get_bool_ids_greater_than(p, limit, return_prob))
        return result
    else:
        result = []
        for i, p in enumerate(probs):
            if p > limit:
                if return_prob:
                    result.append((i, p))
                else:
                    result.append(i)
        return result


def get_span(start_ids, end_ids, with_prob=False):
    """
    Get span set from position start and end list.
    """
    if with_prob:
        start_ids = sorted(start_ids, key=lambda x: x[0])
        end_ids = sorted(end_ids, key=lambda x: x[0])
    else:
        start_ids = sorted(start_ids)
        end_ids = sorted(end_ids)

    start_pointer = 0
    end_pointer = 0
    len_start = len(start_ids)
    len_end = len(end_ids)
    couple_dict = {}
    while start_pointer < len_start and end_pointer < len_end:
        if with_prob:
            start_id = start_ids[start_pointer][0]
            end_id = end_ids[end_pointer][0]
        else:
            start_id = start_ids[start_pointer]
            end_id = end_ids[end_pointer]

        if start_id == end_id:
            couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
            start_pointer += 1
            end_pointer += 1
            continue
        if start_id < end_id:
            couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
            start_pointer += 1
            continue
        if start_id > end_id:
            end_pointer += 1
            continue
    result = [(couple_dict[end], end) for end in couple_dict]
    result = set(result)
    return result


def dbc2sbc(s):
    rs = ""
    for char in s:
        code = ord(char)
        if code == 0x3000:
            code = 0x0020
        else:
            code -= 0xfee0
        if not (0x0021 <= code and code <= 0x7e):
            rs += char
            continue
        rs += chr(code)
    return rs


def cut_chinese_sent(para):
    para = re.sub(r'([。！？\?])([^”’])', r'\1\n\2', para)
    para = re.sub(r'(\.{6})([^”’])', r'\1\n\2', para)
    para = re.sub(r'(\…{2})([^”’])', r'\1\n\2', para)
    para = re.sub(r'([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    para = para.rstrip()
    return para.split("\n")


def get_id_and_prob(span_set, offset_mapping):
    """
    Return text id and probability of predicted spans.
    """
    prompt_end_token_id = offset_mapping[1:].index([0, 0])
    bias = offset_mapping[prompt_end_token_id][1] + 1
    for index in range(1, prompt_end_token_id + 1):
        offset_mapping[index][0] -= bias
        offset_mapping[index][1] -= bias

    sentence_id = []
    prob = []
    for start, end in span_set:
        prob.append(start[1] * end[1])
        start_id = offset_mapping[start[0]][0]
        end_id = offset_mapping[end[0]][1]
        sentence_id.append((start_id, end_id))
    return sentence_id, prob


load_input_dataset = None
load_output_dataset = None
input_data = []
output_data = []
model_desc = 0
run_mode = 0
INDEX = 0

def main():
    global input_data 

    # texts = [
    #     '"北京市海淀区人民法院\n民事判决书\n(199x)建初字第xxx号\n原告：张三。\n委托代理人李四，北京市 A律师事务所律师。\n被告：B公司，法定代表人王五，开发公司总经理。\n委托代理人赵六，北京市 C律师事务所律师。"',
    #     '原告赵六，2022年5月29日生\n委托代理人孙七，深圳市C律师事务所律师。\n被告周八，1990年7月28日出生\n委托代理人吴九，山东D律师事务所律师'
    # ]
    # texts = ["北京市海淀区人民法院\n民事判决书\n(199x)建初字第xxx号\n原告：张三。\n委托代理人李四，北京市 A律师事务所律师。\n被告：B公司，法定代表人王五，开发公司总经理。\n委托代理人赵六，北京市 C律师事务所律师。"
    # ]
    # schema1 = ["法院", {"原告": "委托代理人"}, {"被告": "委托代理人"}]
    # schema2 = [{"原告": ["出生日期", "委托代理人"]}, {"被告": ["出生日期", "委托代理人"]}]
    # texts = ['2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！', '']
    # schema1 = ['时间', '选手', '赛事名称']
    # schema1 = {'竞赛名称': ['主办方', '承办方', '已举办次数']}
    # result_type = 'in'
    texts = ['中国地震台网正式测定：5月16日06时08分在云南临沧市凤庆县(北纬24.34度，东经99.98度)发生3.5级地震，震源深度10千米。']
    schema1 = {'地震触发词': ['地震强度', '时间', '震中位置', '震源深度']}
    predictor = UIEPredictor(512, 1, schema1, position_prob=0.5)
    print("-----------------------------")
    outputs = predictor.predict(texts)
    print(outputs)


if __name__ == '__main__':
    main()