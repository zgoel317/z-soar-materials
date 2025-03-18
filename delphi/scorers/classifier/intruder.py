import asyncio
import json
import re
from dataclasses import dataclass
from typing import Literal, Sequence

from ...clients.client import Client
from ...latents import ActivatingExample, Example, LatentRecord
from ...logger import logger
from .classifier import Classifier, ScorerResult
from .prompts.intruder_prompt import prompt as intruder_prompt
from .sample import _prepare_text


@dataclass
class IntruderSample:
    """
    A sample for an intruder experiment.
    """

    examples: list[str]
    intruder_index: int
    chosen_quantile: int


@dataclass
class IntruderWord(IntruderSample):
    """
    A sample for an intruder word experiment.
    """


@dataclass
class IntruderSentence(IntruderSample):
    """
    A sample for an intruder sentence experiment.
    """

    activations: list[list[float]]
    tokens: list[list[str]]
    intruder_distance: float


@dataclass
class IntruderResult:
    """
    Result of an intruder experiment.
    """

    interpretation: str = ""
    sample: IntruderSample | None = None
    prediction: int = 0
    correct_index: int = -1
    correct: bool = False


class IntruderScorer(Classifier):
    name = "intruder"

    def __init__(
        self,
        client: Client,
        verbose: bool = False,
        n_examples_shown: int = 1,
        log_prob: bool = False,
        temperature: float = 0.0,
        cot: bool = False,
        type: Literal["word", "sentence", "fuzzed"] = "word",
        seed: int = 42,
        **generation_kwargs,
    ):
        """
        Initialize a IntruderScorer.

        Args:
            client: The client to use for generation.
            tokenizer: The tokenizer used to cache the tokens
            verbose: Whether to print verbose output.
            n_examples_shown: The number of examples to show in the prompt,
                        a larger number can both leak information and make
                        it harder for models to generate anwers in the correct format
            log_prob: Whether to use log probabilities to allow for AUC calculation
            temperature: The temperature to use for generation
            type: The type of intruder to use, either "word" or "sentence"
            generation_kwargs: Additional generation kwargs
        """
        super().__init__(
            client=client,
            verbose=verbose,
            n_examples_shown=n_examples_shown,
            log_prob=log_prob,
            temperature=temperature,
            seed=seed,
            **generation_kwargs,
        )
        self.type = type
        self.cot = cot
    def prompt(self, examples: str) -> list[dict]:
        return intruder_prompt(examples,cot=self.cot)

    async def __call__(
        self,
        record: LatentRecord,
    ) -> ScorerResult:
        samples = self._prepare_and_batch(record)

        results = await self._query(
            samples,
        )

        return ScorerResult(record=record, score=results)

    def _count_words(self, examples: Sequence[Example]) -> dict[str, int]:
        """
        Count the number of words in the examples and return a dictionary of the counts.
        If activating examples are provided, count activating tokens.
        If non-activating examples are provided, count activating tokens if they exist,
        otherwise count non-activating tokens.
        """
        counts = {}
        for example in examples:
            str_tokens = example.str_tokens
            if example.normalized_activations is not None:
                acts = example.normalized_activations
                # TODO: this is a hack instead of using a threshold
                # select only acts that are larger than 2
                acts[acts < 2] = 0
            else:
                acts = example.activations
            if acts.max() > 0:
                wanted_indices = acts.nonzero()
                for index in wanted_indices:
                    if str_tokens[index] not in counts:
                        counts[str_tokens[index]] = 0
                    counts[str_tokens[index]] += 1
            else:
                for token in str_tokens:
                    if token not in counts:
                        counts[token] = 0
                    counts[token] += 1
        return counts

    def _prepare(self, record: LatentRecord) -> None:
        pass

    def _get_quantiled_examples(
        self, examples: list[ActivatingExample]
    ) -> dict[int, list[ActivatingExample]]:
        """
        Get the quantiled examples.
        """
        quantiles = {}
        for example in examples:
            if example.quantile not in quantiles:
                quantiles[example.quantile] = []
            quantiles[example.quantile].append(example)
        return quantiles

    def _prepare_and_batch(self, record: LatentRecord) -> list[IntruderSample]:
        """
        Prepare and shuffle a list of samples for classification.
        """

        assert len(record.not_active) > 0, "No non-activating examples found"
        batches = []
        quantiled_intruder_sentences = self._get_quantiled_examples(record.test)
        if self.type == "word":
            # count words
            non_active_counts = self._count_words(record.not_active)
            quantiled_active_counts: dict[int, list[tuple[str, int]]] = {}
            quantile_non_active_counts: dict[int, dict[str, int]] = {}
            number_quantiles = len(quantiled_intruder_sentences)
            for quantiles,examples in quantiled_intruder_sentences.items():
                active_counts = self._count_words(examples)

                # if not more than 3 examples skip
                if len(active_counts) < 3:
                    continue
                # non active words can't be in the active words
                exclusive_non_active_counts = {
                    k: v for k, v in non_active_counts.items() if k not in active_counts
                }
                quantile_non_active_counts[quantiles] = exclusive_non_active_counts
                chosen_active = sorted(
                    active_counts.items(), key=lambda x: x[1], reverse=True
                )
                quantiled_active_counts[quantiles] = chosen_active
            for quantiles,active_counts in quantiled_active_counts.items():
                # for each quantile, do some intruder words
                n_samples = len(record.not_active)//number_quantiles
                non_active_samples = quantile_non_active_counts[quantiles]
                n_samples = min(n_samples,len(non_active_samples))
                
                #randomly choose n_samples from non_active_samples
                #TODO: do we also want it to be weighted by the count?
                non_active_samples = self.rng.sample(list(non_active_samples.keys()),
                                                     n_samples)

                n_active_samples = min(len(active_counts),self.n_examples_shown-1)
                
                # Extract just the words from the (word, count) tuples in active_counts,
                # discarding the count values
                active_examples,active_counts = zip(*active_counts)
                for intruder_word in non_active_samples:
                    # choose the active words, with a frequency weighted by the count
                    # without replacement
                    examples_to_choose = list(active_examples).copy()
                    counts_to_choose = list(active_counts).copy()
                    chosen_examples = []
                    for i in range(n_active_samples):
                        chosen_index = self.rng.choices(range(len(examples_to_choose)),
                                                        weights=counts_to_choose,
                                                        k=1)[0]
                        chosen_examples.append(examples_to_choose[chosen_index])
                        examples_to_choose.pop(chosen_index)
                        counts_to_choose.pop(chosen_index)

                    # chose the index of the intruder
                    intruder_index = self.rng.randint(0, n_active_samples- 1)
                    # add the intruder to the examples
                    examples = chosen_examples.copy()
                    examples.insert(intruder_index, intruder_word)
                    # create the sample
                    batches.append(
                        IntruderWord(
                            examples=examples,
                            intruder_index=intruder_index,
                            chosen_quantile=quantiles,
                        )
                    )

            
        if self.type == "sentence" or self.type == "fuzzed":
            intruder_sentences = record.not_active
            for i, intruder in enumerate(intruder_sentences):
                quantile_index = self.rng.randint(
                    0, len(quantiled_intruder_sentences.keys()) - 1
                )
                active_examples = quantiled_intruder_sentences[quantile_index]
                examples_to_show = min(self.n_examples_shown - 1, len(active_examples))
                example_indices = self.rng.sample(
                    range(len(active_examples)), examples_to_show
                )
                active_examples = [active_examples[i] for i in example_indices]

                # convert the examples to strings
                constructed_examples = []
                intruder_index = self.rng.randint(0, examples_to_show)
                if self.type == "sentence":
                    # no highlighting
                    constructed_examples = [
                        "".join(example.str_tokens) for example in active_examples
                    ]
                    intruder_sentence = "".join(intruder.str_tokens)
                elif self.type == "fuzzed":
                    # highlights the active tokens
                    constructed_examples = []
                    active_tokens = 0
                    for example in active_examples:
                        text, _ = _prepare_text(
                            example, n_incorrect=0, threshold=0.3, highlighted=True
                        )
                        constructed_examples.append(text)
                        active_tokens += (example.activations > 0).sum().item()
                    active_tokens = int(active_tokens / len(active_examples))
                    # if example is contrastive, use the active tokens
                    # otherwise use the non-activating tokens
                    if intruder.activations.max() > 0:
                        n_incorrect = 0
                    else:
                        n_incorrect = active_tokens
                    intruder_sentence, _ = _prepare_text(
                        intruder,
                        n_incorrect=n_incorrect,
                        threshold=0.3,
                        highlighted=True,
                    )

                constructed_examples.insert(intruder_index, intruder_sentence)
                used_sentences = []
                # TODO: have to do this instead of insert because of
                # typing issues :(
                activations = []
                tokens = []
                for i, example in enumerate(active_examples):
                    if i == intruder_index:
                        used_sentences.append(intruder)
                        activations.append(intruder.activations.tolist())
                        tokens.append(intruder.str_tokens)
                        used_sentences.append(example)
                        activations.append(example.activations.tolist())
                        tokens.append(example.str_tokens)
                    else:
                        used_sentences.append(example)
                        activations.append(example.activations.tolist())
                        tokens.append(example.str_tokens)
                batches.append(
                    IntruderSentence(
                        examples=constructed_examples,
                        intruder_index=intruder_index,
                        chosen_quantile=quantile_index,
                        activations=activations,
                        tokens=tokens,
                        intruder_distance=intruder.distance,
                    )
                )

        return batches

    async def _query(
        self,
        samples: list[IntruderSample],
    ) -> list[IntruderResult]:
        """
        Send and gather batches of samples to the model.
        """
        sem = asyncio.Semaphore(1)

        async def _process(sample):
            async with sem:
                result = await self._generate(sample)
                return result

        tasks = [asyncio.create_task(_process(sample)) for sample in samples]
        results = await asyncio.gather(*tasks)

        return results

    def _build_prompt(
        self,
        sample: IntruderSample,
    ) -> list[dict]:
        """
        Prepare prompt for generation.
        """

        examples = "\n".join(
            f"Example {i}: {example}" for i, example in enumerate(sample.examples)
        )

        return self.prompt(examples=examples)

    def _parse(
        self,
        string: str,
    ) -> tuple[str, int]:
        """The answer will be in the format interpretation [RESPONSE]: 1"""
        # Find the first instance of the text with [RESPONSE]:
        pattern = r"\[RESPONSE\]:"
        match = re.search(pattern, string)
        if match is None:
            raise ValueError("No response found in string")
        # get everything before the match
        interpretation = string[: match.start()]
        # get everything after the match
        after = string[match.end() :]
        # the response should be a single number
        try:
            prediction = int(after.strip())
        except ValueError:
            raise ValueError("Response is not a number")
        if prediction < 0 or prediction >= self.n_examples_shown:
            raise ValueError("Response is out of range")
        return interpretation, prediction

    async def _generate(self, sample: IntruderSample) -> IntruderResult:
        """
        Generate predictions for a batch of samples.
        """

        prompt = self._build_prompt(sample)
        try:
            response = await self.client.generate(prompt, **self.generation_kwargs)
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            response = None

        if response is None:
            # default result is a error
            return IntruderResult()
        else:

            try:
                interpretation, prediction = self._parse(response.text)
            except Exception as e:
                logger.error(f"Parsing selections failed: {e}")
                # default result is a error
                return IntruderResult()

        # check that the only prediction is the intruder
        correct = prediction == sample.intruder_index

        result = IntruderResult(
            interpretation=interpretation,
            sample=sample,
            prediction=prediction,
            correct_index=sample.intruder_index,
            correct=correct,
        )

        return result
