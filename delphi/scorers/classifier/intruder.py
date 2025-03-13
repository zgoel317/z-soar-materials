import asyncio
import json
import re
from dataclasses import dataclass, field
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

    frequency: list[int]


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
    predictions: list[bool] = field(default_factory=list)
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
        type: Literal["word", "sentence", "fuzzed"] = "word",
        n_quantiles: int = 10,
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
        self.n_quantiles = n_quantiles

    def prompt(self, examples: str) -> list[dict]:
        return intruder_prompt(examples)

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
        if self.type == "word":
            # count words
            non_active_counts = self._count_words(record.not_active)
            active_counts = self._count_words(record.test)

            # sort active_counts
            chosen_active = sorted(
                active_counts.items(), key=lambda x: x[1], reverse=True
            )
            # get the non_active_counts that are not in the active_counts
            exclusive_non_active_counts = {
                k: v for k, v in non_active_counts.items() if k not in active_counts
            }

            max_number_quantiles = min(
                len(chosen_active) // (self.n_examples_shown - 1), self.n_quantiles
            )
            quantilized_active_counts = []
            for i in range(max_number_quantiles):
                # 0 is most frequent, max_number_quantiles is least frequent

                selected = chosen_active[
                    i
                    * (self.n_examples_shown - 1) : (i + 1)
                    * (self.n_examples_shown - 1)
                ]
                quantile_words, quantile_frequencies = zip(*selected)
                quantilized_active_counts.append(
                    (list(quantile_words), list(quantile_frequencies))
                )

            non_active = len(record.not_active)
            # get as many intruder as non_active if possible
            min_n = min(non_active, len(exclusive_non_active_counts))
            # sort the non_active_counts by value
            non_active = sorted(
                exclusive_non_active_counts.items(), key=lambda x: x[1], reverse=True
            )[:min_n]
            if len(non_active) == 0:
                # This happens with contrastive examples sometimes
                non_active = chosen_active[-5:]
            # get the top min_n
            intruder_words, intruder_frequencies = zip(*non_active)

            for i, intruder in enumerate(intruder_words):
                # choose the quantile of the active examples
                quantile_index = self.rng.randint(0, max_number_quantiles - 1)
                (active_examples, active_frequencies) = quantilized_active_counts[
                    quantile_index
                ]

                # chose the index of the intruder
                intruder_index = self.rng.randint(0, self.n_examples_shown - 1)
                # add the intruder to the examples
                examples = active_examples.copy()

                examples.insert(intruder_index, intruder)
                frequencies = active_frequencies.copy()
                frequencies.insert(intruder_index, intruder_frequencies[i])
                # create the sample
                batches.append(
                    IntruderWord(
                        examples=examples,
                        intruder_index=intruder_index,
                        frequency=frequencies,
                        chosen_quantile=quantile_index,
                    )
                )

        if self.type == "sentence" or self.type == "fuzzed":
            intruder_sentences = record.not_active
            quantiled_intruder_sentences = self._get_quantiled_examples(record.test)
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
    ) -> tuple[str, list[bool]]:
        """The answer will be in the format interpretation [RESPONSE]: [0, 1, 0, 1]"""
        # Find the first instance of the text with [RESPONSE]:
        pattern = r"\[RESPONSE\]:"
        match = re.search(pattern, string)
        if match is None:
            raise ValueError("No response found in string")
        # get everything before the match
        interpretation = string[: match.start()]
        # get everything after the match
        after = string[match.end() :]
        # the response is a list so you can pattern it again
        pattern = r"\[.*?\]"
        match = re.search(pattern, after)
        if match is None:
            raise ValueError("List not found")
        response = match.group(0)
        # convert the response to a list
        predictions = json.loads(response)
        # assert the length of the predictions is the same as the number of examples
        assert len(predictions) == self.n_examples_shown
        # return the interpretation and the predictions
        return interpretation, predictions

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
                interpretation, predictions = self._parse(response.text)
            except Exception as e:
                logger.error(f"Parsing selections failed: {e}")
                # default result is a error
                return IntruderResult()

        # check that the only prediction is the intruder
        correct = False
        if sum(predictions) != 1:
            correct = False
        else:
            correct = predictions[sample.intruder_index]

        result = IntruderResult(
            interpretation=interpretation,
            sample=sample,
            predictions=predictions,
            correct_index=sample.intruder_index,
            correct=correct,
        )

        return result
