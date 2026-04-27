import os
import time
import random

from huggingface_hub import InferenceClient
from together import Together
from together.error import RateLimitError, TogetherException

try:
    from together.error import ServiceUnavailableError
except ImportError:
    ServiceUnavailableError = TogetherException


HF_MODEL_DEFAULT = "meta-llama/Llama-3.1-8B-Instruct"
TOGETHER_MODEL_DEFAULT = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"


def _choice_content(choice):
    msg = getattr(choice, "message", None)
    if msg is None and isinstance(choice, dict):
        msg = choice.get("message")
    content = None
    if msg is not None:
        if isinstance(msg, dict):
            content = msg.get("content")
        else:
            content = getattr(msg, "content", None)
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    parts.append(part)
            content = "".join(parts)
    return content or ""


class _SpacedCallLimiter:
    def __init__(self, min_interval_seconds: float):
        self.min_interval = float(min_interval_seconds)
        self._last = 0.0

    def wait(self):
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.monotonic()


class UnifiedLLM:
    def __init__(self):
        hf_model_or_url = (
            os.getenv("HF_ENDPOINT_URL")
            or os.getenv("HF_MODEL")
            or HF_MODEL_DEFAULT
        ).strip()
        hf_token = (os.getenv("HF_TOKEN") or "").strip()
        self.hf_client = InferenceClient(model=hf_model_or_url, token=hf_token, timeout=300)

        self.together = None
        self.together_model = (os.getenv("TOGETHER_MODEL") or TOGETHER_MODEL_DEFAULT).strip()
        tg_key = (os.getenv("TOGETHER_API_KEY") or "").strip()
        if tg_key:
            self.together = Together(api_key=tg_key)
            self._tg_limiter = _SpacedCallLimiter(min_interval_seconds=100.0)

    def _messages_to_prompt(self, messages):
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"<|{role}|>\n{content}\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)

    def _hf_chat(self, messages, max_tokens=512, temperature=0.3):
        tries, delay = 3, 2.5
        last_err = None
        for _ in range(tries):
            try:
                if hasattr(self.hf_client, "chat_completion"):
                    resp = self.hf_client.chat_completion(
                        messages=messages, max_tokens=max_tokens, temperature=temperature, stream=False
                    )
                    return _choice_content(resp.choices[0])
                prompt = self._messages_to_prompt(messages)
                return self.hf_client.text_generation(
                    prompt, max_new_tokens=max_tokens, temperature=temperature, stream=False, return_full_text=False
                )
            except Exception as exc:
                last_err = exc
                time.sleep(delay)
                delay *= 1.8
        raise last_err

    def chat(self, messages, temperature=0.3, max_tokens=512, stream=False):
        try:
            return self._hf_chat(messages, max_tokens=max_tokens, temperature=temperature)
        except Exception:
            if self.together is None:
                raise
            self._tg_limiter.wait()
            backoff = 12.0
            for attempt in range(4):
                try:
                    resp = self.together.chat.completions.create(
                        model=self.together_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=stream,
                    )
                    return _choice_content(resp.choices[0])
                except (RateLimitError, ServiceUnavailableError):
                    if attempt == 3:
                        raise
                    time.sleep(backoff + random.uniform(0, 3))
                    backoff *= 1.8
