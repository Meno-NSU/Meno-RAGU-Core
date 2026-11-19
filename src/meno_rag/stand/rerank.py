import json
import math
from typing import Any

from nltk.stem.snowball import SnowballStemmer

from meno_rag.stand.abbreviation_cache import AbbreviationTokenCache
from meno_rag.stand.prompts import SYSTEM_PROMPT_FOR_RELEVANCE
from meno_rag.stand.rewriting import find_candidates_to_abbreviations

__all__ = [
    "MAX_RERANKER_TOKENS",
    "POSSIBLE_LABELS",
    "RERANK_PROMPT_ORDERS",
    "SYSTEM_PROMPT_FOR_RELEVANCE",
    "aggregate_logprobs_to_score",
    "build_prompt",
    "rerank_merge_score",
    "response_format_schema",
    "score_from_json_response",
    "score_from_logprobs",
]

MAX_RERANKER_TOKENS: int = 8192
POSSIBLE_LABELS = ["0", "1", "2"]


RERANK_PROMPT_ORDERS = ("legacy", "question_first")


def build_prompt(
    user_question: str,
    dialogue_history: str,
    estimated_document: str,
    abbr_dict: dict[str, dict[str, str | list[str]]],
    stemmer: SnowballStemmer | None = None,
    is_json: bool = False,
    order: str = "legacy",
    *, abbreviation_cache: AbbreviationTokenCache | None = None,
) -> list[dict[str, str]]:
    """Build the reranker prompt that scores a candidate document by its USEFULNESS
    as a factual basis for answering the user's question (with dialogue history and
    the abbreviation dictionary), mirroring the QA prompt structure — rather than
    abstract topical relevance to a search query.

    ``order`` (RERANK_PROMPT_ORDER): "legacy" — the abbreviation dictionary, the document, the history, the question,
    the instruction, as the prompts always were; "question_first" — the history, the question and the instruction
    before the dictionary and the document. The sections and their texts are the same; in the second order the part a
    turn's ~85 candidates share comes first, so vLLM's prefix cache computes it once (speed audit of 19.11, П1: 701
    -> 410 new tokens per call on 2,040 real prompts)."""
    if not estimated_document:
        raise ValueError("The estimated document is empty.")
    if order not in RERANK_PROMPT_ORDERS:
        raise ValueError(f"Unknown rerank prompt order {order!r}; expected one of {RERANK_PROMPT_ORDERS}.")
    if abbreviation_cache is not None and abbreviation_cache.stemmer is stemmer:
        found_abbreviations = abbreviation_cache.find(estimated_document, dialogue_history, user_question, abbr_dict)
    else:
        found_abbreviations = find_candidates_to_abbreviations(
            source_text=estimated_document + "\n\n" + dialogue_history + "\n\n" + user_question,
            all_abbreviations=abbr_dict,
            stemmer=stemmer,
        )
    # Each section without the blank line after it; the prompt joins them with one, which is the text the sections
    # used to be added with one by one (tests/test_rerank_prompt_order.py compares it with the old function).
    abbreviation_section = ""
    if found_abbreviations:
        abbreviation_section += "==========\nABBREVIATION DICTIONARY\n==========\n\n"
        abbreviation_section += (
            "**Природа данных:** предположительные расшифровки, подобранные автоматически"
            " по совпадению буквенной формы. Записи могут быть лишними, ошибочными или "
            "неоднозначными (омонимия). Используйте словарь как подсказку, а не как "
            "достоверный источник.\n\n"
        )
        abbreviation_section += "**Записи:**\n" + found_abbreviations
    document_section = "==========\nDOCUMENT 1\n==========\n\n" + estimated_document.strip()
    history_section = ("==========\nDIALOGUE HISTORY\n==========\n\n" + dialogue_history) if dialogue_history else ""
    question_section = "==========\nCURRENT QUESTION\n==========\n\n" + user_question
    instruction_section = "==========\nINSTRUCTION\n==========\n\n"
    instruction_section += (
        "Оцените, пожалуйста, возможность ответа на вопрос из раздела CURRENT QUESTION, "
        "опираясь только на факты из раздела DOCUMENT 1."
    )
    if dialogue_history:
        instruction_section += (
            "\n\nРаздел DIALOGUE HISTORY используйте исключительно для разрешения местоимений, "
            "уточнений и восстановления темы — не извлекайте из него фактические сведения."
        )
    if found_abbreviations:
        instruction_section += "\n\nРаздел ABBREVIATION DICTIONARY используйте по следующим правилам:\n"
        instruction_section += (
            "1. Применяйте расшифровку из словаря, только если сокращение действительно встречается в CURRENT QUESTION"
        )
        instruction_section += ", DIALOGUE HISTORY или DOCUMENT 1" if dialogue_history else " или DOCUMENT 1"
        instruction_section += " и по контексту является аббревиатурой.\n"
        instruction_section += (
            "2. Если одной аббревиатуре соответствует несколько расшифровок (омонимия), "
            "выберите ту, которая согласуется с темой диалога и содержанием документов.\n"
        )
        instruction_section += (
            "3. Если в тексте встречается буквосочетание, совпадающее с ключом словаря, "
            "но по контексту аббревиатурой не является (например, это команда, междометие, "
            "имя собственное, часть слова), — игнорируйте запись словаря.\n"
        )
        instruction_section += (
            "4. Если подходящая расшифровка в словаре отсутствует, а в документах её тоже нет, — "
            "не придумывайте расшифровку; используйте сокращение как есть.\n"
        )
        instruction_section += (
            "5. Словарь не является источником фактов: сам факт наличия записи «X — Y» не доказывает, "
            "что Y упомянут в документах или релевантен вопросу."
        )
    if order == "legacy":
        sections = [abbreviation_section, document_section, history_section, question_section, instruction_section]
    else:
        sections = [history_section, question_section, instruction_section, abbreviation_section, document_section]
    user_prompt = "\n\n".join(section for section in sections if section)
    if is_json:
        modified_chapter_name = "## Формат ответа"
        found_idx = SYSTEM_PROMPT_FOR_RELEVANCE.find(modified_chapter_name)
        if found_idx < 0:
            raise RuntimeError(f"The system prompt does not contain the chapter `{modified_chapter_name}`.")
        modified_system_prompt = SYSTEM_PROMPT_FOR_RELEVANCE[:found_idx] + modified_chapter_name + "\n\n"
        modified_system_prompt += (
            'Ответьте в формате JSON вида `{"label": <your_label>}`, где '
            "`<your_label>` — это цифра 2, 1 или 0. Не добавляйте никаких пояснений, "
            "комментариев или дополнительного текста.\n"
        )
        return [{"role": "system", "content": modified_system_prompt}, {"role": "user", "content": user_prompt}]
    return [{"role": "system", "content": SYSTEM_PROMPT_FOR_RELEVANCE}, {"role": "user", "content": user_prompt}]


def aggregate_logprobs_to_score(label_logprobs: dict[int, float]) -> float:
    """Smooth relevance score in [0, 1] from per-label logprobs over {0, 1, 2}.

    Instead of "P(label 2) or else 0", aggregate the full distribution into the
    expected label E[label] = Σ label·P(label) and normalise by 2. Partial-relevance
    (label 1) chunks therefore get ~0.5 and survive the rerank cut, while label-0
    chunks fall toward 0 — a smoother, less brittle ranking.
    """
    if set(label_logprobs.keys()) != {0, 1, 2}:
        raise ValueError(f"Expected labels {{0, 1, 2}}, got {set(label_logprobs.keys())}.")
    max_lp = max(label_logprobs.values())
    exp_values = {label: math.exp(lp - max_lp) for label, lp in label_logprobs.items()}
    total = sum(exp_values.values())
    probs = {label: exp_val / total for label, exp_val in exp_values.items()}
    expected_label = sum(label * probs[label] for label in probs)
    return expected_label / 2.0


def score_from_logprobs(choice: dict[str, Any]) -> float:
    try:
        top_logprobs_list = choice["logprobs"]["content"][0]["top_logprobs"]
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("Reranker response does not contain top_logprobs.") from exc

    label_logprobs: dict[int, list[float]] = {int(label): [] for label in POSSIBLE_LABELS}
    for item in top_logprobs_list:
        token = str(item.get("token", ""))
        if token not in POSSIBLE_LABELS:
            continue
        value = float(item["logprob"])
        if math.isnan(value) or value == math.inf:
            raise ValueError("Reranker returned a nonfinite label logprob.")
        label_logprobs[int(token)].append(value)
    if not any(math.isfinite(value) for values in label_logprobs.values() for value in values):
        raise ValueError("Reranker top_logprobs contain no finite digit label.")

    raw = {}
    for label, values in label_logprobs.items():
        if not values:
            raw[label] = -100.0
            continue
        peak = max(values)
        # Distinct token IDs can decode to the same digit. Their probability masses add.
        raw[label] = peak if peak == -math.inf else peak + math.log(sum(math.exp(value - peak) for value in values))
    return aggregate_logprobs_to_score(raw)


def score_from_json_response(content: str) -> float:
    """Parse the JSON-fallback label and map it onto the SAME [0, 1] scale as
    `score_from_logprobs` (0 → 0.0, 1 → 0.5, 2 → 1.0) via a one-hot distribution,
    so the two paths are interchangeable downstream."""
    parsed = json.loads(content.strip())
    predicted_label = int(parsed["label"])
    if predicted_label not in {0, 1, 2}:
        raise ValueError(f"Unexpected label: {predicted_label}")
    one_hot = {label: (0.0 if label == predicted_label else -100.0) for label in (0, 1, 2)}
    return aggregate_logprobs_to_score(one_hot)


def rerank_merge_score(retrieval_score: float, rerank_score: float, alpha: float) -> float:
    if rerank_score > 0.0:
        return (1.0 - alpha) * retrieval_score + alpha * rerank_score
    return rerank_score


def response_format_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "relevance",
            "schema": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "enum": ["0", "1", "2"],
                    }
                },
                "required": ["label"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
