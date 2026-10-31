"""Zero-shot classification with a Mapika decider model (optional, needs `decider-ai[gguf]`)."""

from __future__ import annotations

from .base import Example, Prediction
from .normalize import model_text

DEFAULT_REPO = "Mapika/decider-4b-GGUF"
DEFAULT_FILE = "decider-4b-v3-Q4_K_M.gguf"


class DeciderClassifier:
    """Typed single-choice decision over the category names; ignores history."""

    name = "decider-4b"

    def __init__(
        self,
        repo: str = DEFAULT_REPO,
        gguf_file: str | None = DEFAULT_FILE,
        descriptions: dict[str, str] | None = None,
    ) -> None:
        self.repo = repo
        self.gguf_file = gguf_file
        self.descriptions = descriptions or {}
        self._decider = None

    def fit(self, history: list[Example]) -> None:
        pass

    def _load(self):
        if self._decider is None:
            try:
                from decider.infer import Decider
            except ImportError as exc:
                raise RuntimeError("install the model extra: uv sync --group models (decider-ai[gguf])") from exc
            self._decider = Decider(self.repo, gguf_file=self.gguf_file) if self.gguf_file else Decider(self.repo)
        return self._decider

    def _question(self, categories: list[str]) -> str:
        lines = ["Which budget category does this bank transaction belong to?"]
        lines += [f"- {c}: {self.descriptions[c]}" for c in categories if c in self.descriptions]
        return "\n".join(lines)

    def predict(self, example: Example, categories: list[str]) -> Prediction:
        state = "Bank transaction (German): " + model_text(example.payee, example.purpose)
        answer = self._load().decide(state, [{"question": self._question(categories), "options": categories}])[0]
        probs = {k: float(v) for k, v in answer.get("probs", {}).items() if k in categories}
        choice = answer.get("choice")
        if choice not in categories:
            return Prediction(None)
        return Prediction(choice, float(answer.get("confidence", probs.get(choice, 0.0))), probs)
