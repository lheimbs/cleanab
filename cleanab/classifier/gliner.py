"""Zero-shot classification with Fastino GLiNER2 (optional, needs the `gliner2` package)."""

from __future__ import annotations

from typing import Any

from .base import Example, Prediction
from .normalize import model_text

DEFAULT_MODEL = "fastino/gliner2.5-multi-decide"


def parse_result(result: Any, labels: list[str]) -> tuple[str | None, float, dict[str, float]]:
    """Normalize the shapes classify_text may return into (label, confidence, scores).

    Accepted: {"task": "label"}, {"task": {"label": ..., "confidence": ...}}, {"task": [{"label":..}, ..]}
    or {"task": {"label": score, ...}}. Unknown shapes yield no prediction instead of a guess.
    """
    if isinstance(result, dict) and len(result) == 1:
        result = next(iter(result.values()))
    if isinstance(result, str):
        return (result, 1.0, {result: 1.0}) if result in labels else (None, 0.0, {})
    if isinstance(result, dict):
        if "label" in result:
            label = result["label"]
            confidence = float(result.get("confidence", result.get("score", 1.0)))
            return (label, confidence, {label: confidence}) if label in labels else (None, 0.0, {})
        scores = {k: float(v) for k, v in result.items() if k in labels and isinstance(v, int | float)}
    elif isinstance(result, list):
        scores = {}
        for item in result:
            if isinstance(item, dict) and item.get("label") in labels:
                scores[item["label"]] = float(item.get("confidence", item.get("score", 1.0)))
            elif isinstance(item, str) and item in labels:
                scores[item] = 1.0
    else:
        return None, 0.0, {}
    if not scores:
        return None, 0.0, {}
    best = max(scores, key=scores.__getitem__)
    return best, scores[best], scores


class GlinerClassifier:
    """Zero-shot: ignores history, reads the category names (and optional descriptions) directly."""

    name = "gliner2"

    def __init__(self, model: str = DEFAULT_MODEL, descriptions: dict[str, str] | None = None) -> None:
        self.model_id = model
        self.descriptions = descriptions or {}
        self._extractor = None

    def fit(self, history: list[Example]) -> None:
        pass

    def _load(self):
        if self._extractor is None:
            try:
                from gliner2 import GLiNER2
            except ImportError as exc:
                raise RuntimeError(
                    'install the model extra: uv sync --group models (pip install "gliner2[local]")'
                ) from exc
            self._extractor = GLiNER2.from_pretrained(self.model_id)
        return self._extractor

    def _label_text(self, category: str) -> str:
        description = self.descriptions.get(category)
        return f"{category}: {description}" if description else category

    def predict(self, example: Example, categories: list[str]) -> Prediction:
        by_text = {self._label_text(c): c for c in categories}
        result = self._load().classify_text(model_text(example.payee, example.purpose), {"category": list(by_text)})
        label, confidence, scores = parse_result(result, list(by_text))
        if label is None:
            return Prediction(None)
        return Prediction(by_text[label], confidence, {by_text[k]: v for k, v in scores.items()})
