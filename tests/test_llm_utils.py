import importlib.util
import sys
import types
import unittest
from unittest.mock import patch


if importlib.util.find_spec("huggingface_hub") is None:
    huggingface_hub = types.ModuleType("huggingface_hub")
    huggingface_hub.InferenceClient = object
    sys.modules["huggingface_hub"] = huggingface_hub

if importlib.util.find_spec("together") is None:
    together = types.ModuleType("together")
    together.Together = object
    together_error = types.ModuleType("together.error")
    together_error.TogetherException = type("TogetherException", (Exception,), {})
    together_error.RateLimitError = type(
        "RateLimitError", (together_error.TogetherException,), {}
    )
    sys.modules["together"] = together
    sys.modules["together.error"] = together_error

from llm_utils import UnifiedLLM


class _AlwaysFailingClient:
    def __init__(self):
        self.calls = 0
        self.errors = [RuntimeError(f"simulated provider failure {i}") for i in range(3)]

    def chat_completion(self, **_kwargs):
        error = self.errors[self.calls]
        self.calls += 1
        raise error


class UnifiedLLMTests(unittest.TestCase):
    def test_hf_chat_does_not_sleep_after_final_attempt(self):
        client = UnifiedLLM.__new__(UnifiedLLM)
        provider = _AlwaysFailingClient()
        client.hf_client = provider

        with patch("llm_utils.time.sleep") as sleep:
            with self.assertRaises(RuntimeError) as raised:
                client._hf_chat([{"role": "user", "content": "local fixture"}])

        self.assertEqual(provider.calls, 3)
        self.assertIs(raised.exception, provider.errors[-1])
        waits = [call.args[0] for call in sleep.call_args_list]
        self.assertEqual(waits, [2.5, 4.5])


if __name__ == "__main__":
    unittest.main()
