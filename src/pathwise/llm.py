"""One structured-output call to Claude, through the Anthropic API or the Claude Code CLI.

The CLI backend (PATHWISE_BACKEND=claude-code) runs the same prompt and schema
through `claude -p`, so a Claude subscription can power local use and recordings
instead of API credits.
"""

import json
import os
import shutil
import subprocess
import tempfile

import anthropic

MODEL = os.environ.get("PATHWISE_MODEL", "claude-opus-5-5")
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, client=None, model: str = MODEL, backend: str | None = None):
        backend = backend or os.environ.get("PATHWISE_BACKEND", "api")
        self.model = model
        self.cli = None
        if client is None and backend == "claude-code":
            self.cli = os.environ.get("PATHWISE_CLAUDE_BIN") or shutil.which("claude")
            if not self.cli:
                raise LLMError("Claude Code CLI not found. Set PATHWISE_CLAUDE_BIN to its path.")
        self.client = client or (None if self.cli else anthropic.Anthropic())

    def parse(self, *, system: str, prompt: str, output_format, effort: str = "medium"):
        return self._cli(system, prompt, output_format, effort) if self.cli else self._api(system, prompt, output_format, effort)

    def _api(self, system, prompt, output_format, effort):
        response = self.client.beta.messages.parse(
            model=self.model,
            max_tokens=16000,
            betas=[FALLBACK_BETA],
            fallbacks="default",
            output_config={"effort": effort},
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_format=output_format,
        )
        if response.stop_reason == "refusal":
            raise LLMError("Claude declined this request.")
        if response.stop_reason == "max_tokens":
            raise LLMError("The answer was cut off at the token limit.")
        if getattr(response, "parsed_output", None) is None:
            raise LLMError("Claude's response did not match the expected format.")
        return response.parsed_output

    def _cli(self, system, prompt, output_format, effort):
        cmd = [self.cli, "-p", "--output-format", "json", "--json-schema", json.dumps(output_format.model_json_schema()),
               "--system-prompt", system, "--model", self.model, "--effort", effort,
               "--tools", "", "--setting-sources", "", "--no-session-persistence"]
        with tempfile.TemporaryDirectory() as cwd:  # keeps project files and CLAUDE.md out of the prompt
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=600, cwd=cwd)
        try:
            out = json.loads(proc.stdout)
        except json.JSONDecodeError:
            raise LLMError(f"Claude Code CLI failed: {(proc.stderr or proc.stdout).strip()[:300]}")
        if out.get("stop_reason") == "refusal":
            raise LLMError("Claude declined this request.")
        if out.get("is_error") or out.get("structured_output") is None:
            raise LLMError(f"Claude Code CLI returned no structured output: {str(out.get('result'))[:300]}")
        return output_format.model_validate(out["structured_output"])
