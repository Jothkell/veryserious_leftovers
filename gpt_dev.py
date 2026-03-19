#!/usr/bin/env python3
"""
A minimal terminal tool that lets an OpenAI model (e.g., gpt-4o or gpt-5 if available
in your account) propose file edits and shell commands — and then executes them.

Default = safe/interactive: shows diffs and commands, asks for confirmation.
Optional --unsafe = auto-apply edits & run commands without prompting (DANGEROUS).

It also supports an aliasable interactive mode (`gpt`) where you:
  - Run `gpt` (no quotes). A new line opens for freeform, multi-line input.
  - You can paste large text; the script **includes the full text** but only *echoes* it as
    `[pasted N characters]` for any single line > 30 chars.
  - You can drag-and-drop image files into the terminal; paths are detected and attached
    to the model as inline vision inputs.

Usage examples:
  python gpt_dev.py "Initialize a FastAPI service with /healthz and a Dockerfile" --root .
  python gpt_dev.py --interactive --root .
  python gpt_dev.py --interactive --unsafe --model gpt-4o

To alias as `gpt` (bash/zsh):
  alias gpt='python /ABS/PATH/gpt_dev.py --interactive --root .'

Dependencies:
  pip install --upgrade openai

Notes:
- We intentionally keep the OpenAI call to Chat Completions for widest compatibility.
- If you have access to newer models (e.g., gpt-5), pass --model accordingly.
- Never run with --unsafe outside a sandbox. You are giving the model power to modify files
  and execute arbitrary shell commands.
"""
import argparse
import base64
import difflib
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# --------------------------- Configuration ---------------------------
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # change if you have gpt-5 access
MAX_TREE_FILES = int(os.getenv("GPTDEV_MAX_TREE_FILES", "80"))
MAX_SAMPLED_BYTES = int(os.getenv("GPTDEV_MAX_SAMPLED_BYTES", "40000"))
PASTE_ECHO_THRESHOLD = 30  # chars; lines longer than this are echoed as [pasted N characters]
DENYLIST = [
    r"rm\s+-rf\s+/$",
    r"rm\s+-rf\s+/\s",
    r"mkfs",
    r"diskutil\s+erase",
    r"shutdown\\b",
    r"reboot\\b",
    r":\(\)\{\s*:\|:\s*;\s*\}&:\$",  # fork bomb pattern-ish
]
ALLOW_DANGEROUS = os.getenv("GPTDEV_ALLOW_DANGEROUS", "0") == "1"

SYSTEM_PROMPT = (
    "You are a meticulous, benevolent senior software engineer working strictly via structured actions.\n"
    "You may only respond with a JSON object that matches the schema below.\n"
    "Goal: The user wants you to modify files in a local repo and run shell commands to accomplish the task.\n"
    "Rules:\n"
    "- Always include commands needed to set up or verify (e.g., install deps, run tests).\n"
    "- Use portable bash commands that work on macOS.\n"
    "- Never start background servers that block the terminal; use short-lived commands.\n"
    "- File edits should include full desired file content (idempotent).\n"
    "- Keep explanations concise.\n"
    "Schema fields: edits[], commands[], explanation (string).\n"
)

SCHEMA_DOC = {
    "type": "object",
    "properties": {
        "edits": {
            "type": "array",
            "description": "List of file edit operations to apply",
            "items": {
                "type": "object",
                "required": ["path", "action", "content"],
                "properties": {
                    "path": {"type": "string", "description": "relative path under the repo root"},
                    "action": {"type": "string", "enum": ["write", "replace", "append"]},
                    "content": {"type": "string", "description": "entire file content or appended chunk"}
                },
            },
            "default": [],
        },
        "commands": {
            "type": "array",
            "description": "Shell commands to run sequentially (bash)",
            "items": {
                "type": "object",
                "required": ["run"],
                "properties": {
                    "run": {"type": "string"},
                    "timeout_s": {"type": "integer", "default": 180},
                    "cwd": {"type": "string", "description": "optional subdir to run in"}
                },
            },
            "default": [],
        },
        "explanation": {"type": "string"},
    },
    "required": ["edits", "commands"],
    "additionalProperties": False,
}

USER_TEMPLATE = (
    "You have the following workspace snapshot. Propose the necessary edits and commands to achieve the user's instruction.\n\n"
    "# USER INSTRUCTION\n{goal}\n\n"
    "# WORKSPACE TREE\n{tree}\n\n"
    "# FILE SAMPLES (truncated)\n{samples}\n\n"
    "Respond ONLY with JSON, no markdown, no code fences."
)

# --------------------------- Helpers ---------------------------

def inside_root(root: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def denylisted(cmd: str) -> bool:
    for pat in DENYLIST:
        if re.search(pat, cmd):
            return True
    return False


def read_text_safe(p: Path, limit: int) -> str:
    try:
        with p.open("r", errors="replace") as f:
            data = f.read(limit)
        if p.stat().st_size > limit:
            data += "\n... [truncated] ...\n"
        return data
    except Exception as e:
        return f"<unreadable: {e}>\n"


def snapshot_workspace(root: Path, max_files: int, max_bytes: int) -> Dict[str, str]:
    files = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and not any(seg.startswith(".") and seg not in {".env", ".gitignore"} for seg in p.parts):
            files.append(p)
        if len(files) >= max_files:
            break
    tree_lines = ["." ]
    for p in files:
        tree_lines.append(str(p.relative_to(root)))
    tree = "\n".join(tree_lines)

    samples = []
    budget = max_bytes
    for p in files:
        if budget <= 0:
            break
        rel = str(p.relative_to(root))
        head = read_text_safe(p, min(4096, budget))
        budget -= len(head)
        samples.append(f"=== {rel} ===\n{head}")
    return {"tree": tree, "samples": "\n\n".join(samples)}


def confirm(prompt: str) -> bool:
    ans = input(f"{prompt} [y/N]: ").strip().lower()
    return ans in {"y", "yes"}


def show_diff(old: str, new: str, path: str) -> str:
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    return "".join(diff)


def atomic_write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent)) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def path_is_image(p: Path) -> bool:
    mime, _ = mimetypes.guess_type(str(p))
    return bool(mime and mime.startswith("image/"))


def encode_image_data_url(p: Path) -> str:
    mime, _ = mimetypes.guess_type(str(p))
    if not mime:
        mime = "application/octet-stream"
    b = p.read_bytes()
    b64 = base64.b64encode(b).decode("ascii")
    return f"data:{mime};base64,{b64}"


def summarize_for_echo(text: str) -> str:
    # Collapse any single line longer than threshold into a placeholder for echo purposes only.
    lines = text.splitlines()
    out = []
    for line in lines:
        if len(line) > PASTE_ECHO_THRESHOLD:
            out.append(f"[pasted {len(line)} characters]")
        else:
            out.append(line)
    return "\n".join(out)


def read_instruction_interactive() -> Tuple[str, List[Path]]:
    print("Enter your instruction. Finish with Ctrl-D (Linux/macOS) or Ctrl-Z, Enter (Windows).\n"
          "Tip: drag-and-drop image files here to attach them.")
    buf = sys.stdin.read()
    if not buf.strip():
        print("[error] No instruction received. Exiting without calling the API.")
        sys.exit(1)
    # Detect file paths that exist and are images; keep others as text
    possible_paths = re.findall(r"(?P<path>(?:/|~)[^\s]+)", buf)
    images: List[Path] = []
    for raw in possible_paths:
        p = Path(os.path.expanduser(raw))
        if p.exists() and p.is_file() and path_is_image(p):
            images.append(p)
    echo = summarize_for_echo(buf)
    print("\n[Input preview]\n" + (echo.strip() or "(empty)"))
    if images:
        for p in images:
            print(f"[attached image] {p}")
    return buf, images

# --------------------------- OpenAI ---------------------------
try:
    from openai import OpenAI
except Exception as e:
    print("\nERROR: openai package missing. Run: pip install openai\n", file=sys.stderr)
    raise


_OPENAI_CLIENT: OpenAI | None = None


def _load_openai_client() -> OpenAI:
    """Return an OpenAI client using the OPENAI_API_KEY env var."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set OPENAI_API_KEY in your environment before running gpt_dev.py"
        )
    return OpenAI(api_key=api_key)


def _get_openai_client() -> OpenAI:
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        _OPENAI_CLIENT = _load_openai_client()
    return _OPENAI_CLIENT


def llm_plan(model: str, goal: str, root: Path, images: List[Path] | None = None) -> Dict[str, Any]:
    client = _get_openai_client()
    snap = snapshot_workspace(root, MAX_TREE_FILES, MAX_SAMPLED_BYTES)

    # Build a multimodal content array if images are present
    content: List[Dict[str, Any]] = [{"type": "text", "text": USER_TEMPLATE.format(goal=goal, tree=snap["tree"], samples=snap["samples"])}]
    if images:
        for p in images:
            try:
                data_url = encode_image_data_url(p)
                content.append({
                    "type": "input_image",
                    "image_url": {"url": data_url}
                })
            except Exception as e:
                print(f"[warn] could not attach image {p}: {e}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": "JSON Schema:"},
        {"role": "system", "content": json.dumps(SCHEMA_DOC)},
        {"role": "user", "content": content},
    ]

    # We ask for strict JSON. If the model leaks extra tokens, we try to extract the JSON.
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        # If your model supports it, uncomment to force JSON:
        # response_format={"type": "json_object"},
    )
    content_resp = resp.choices[0].message.content

    def parse_json_maybe(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except Exception:
            # Greedy JSON extraction
            m = re.search(r"\{[\s\S]*\}$", text.strip())
            if m:
                return json.loads(m.group(0))
            raise

    plan = parse_json_maybe(content_resp)
    # basic validation
    if not isinstance(plan, dict) or "edits" not in plan or "commands" not in plan:
        raise ValueError("Model response missing required keys: edits, commands")
    return plan


# --------------------------- Executors ---------------------------

def apply_edits(edits: List[Dict[str, Any]], root: Path, unsafe: bool):
    for i, e in enumerate(edits, 1):
        path = root / e.get("path", "")
        action = e.get("action")
        content = e.get("content", "")
        if not path or not action:
            print(f"[edit {i}] skipped: missing path/action")
            continue
        if not inside_root(root, path):
            print(f"[edit {i}] SKIP (outside root): {path}")
            continue

        old = ""
        if path.exists():
            try:
                old = path.read_text()
            except Exception:
                old = "<binary or unreadable>\n"
        new = content if action in {"write", "replace"} else (old + content)

        diff = show_diff(old, new, str(path))
        if not unsafe:
            print(f"\n--- Proposed edit {i}/{len(edits)}: {path}\n{diff if diff else '(no diff)'}")
            if not confirm("Apply this edit?"):
                print("  -> skipped")
                continue
        else:
            print(f"\n[unsafe] Applying edit {i}/{len(edits)}: {path}")
        atomic_write(path, new)
        print("  -> done")


def run_commands(cmds: List[Dict[str, Any]], root: Path, unsafe: bool):
    for i, c in enumerate(cmds, 1):
        run = c.get("run")
        if not run:
            continue
        if denylisted(run) and not (unsafe and ALLOW_DANGEROUS):
            print(f"\n[command {i}] BLOCKED by denylist: {run}")
            continue
        timeout = int(c.get("timeout_s", 180))
        rel_cwd = c.get("cwd")
        cwd = root / rel_cwd if rel_cwd else root
        if not inside_root(root, cwd):
            print(f"[command {i}] SKIP (cwd outside root): {cwd}")
            continue

        if not unsafe:
            print(f"\n--- Proposed command {i}/{len(cmds)} (cwd={cwd}):\n$ {run}")
            if not confirm("Run this command?"):
                print("  -> skipped")
                continue
        else:
            print(f"\n[unsafe] Running command {i}/{len(cmds)} (cwd={cwd}):\n$ {run}")

        try:
            res = subprocess.run(run, shell=True, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
            print("  -> exit", res.returncode)
            if res.stdout:
                print(res.stdout)
            if res.stderr:
                print(res.stderr, file=sys.stderr)
        except subprocess.TimeoutExpired:
            print(f"  -> timed out after {timeout}s")


# --------------------------- CLI ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Let an OpenAI model edit files & execute commands for a given goal.")
    ap.add_argument("goal", nargs="?", help="High-level instruction, or omit to use --interactive")
    ap.add_argument("--root", default=".", help="Project root (restricted sandbox)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Model ID, e.g., gpt-4o or gpt-5 if available")
    ap.add_argument("--unsafe", action="store_true", help="Auto-apply edits & run commands without prompts (DANGEROUS)")
    ap.add_argument("--interactive", action="store_true", help="Read multi-line instruction from stdin; detect dropped image paths")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"Root does not exist: {root}", file=sys.stderr)
        sys.exit(2)

    if args.unsafe:
        print("WARNING: --unsafe grants the model permission to modify files and run arbitrary shell commands.\n"
              "Use in a disposable repo or VM. Set GPTDEV_ALLOW_DANGEROUS=1 to bypass the simple denylist.")

    images: List[Path] = []
    goal = args.goal
    if args.interactive or goal is None:
        goal, images = read_instruction_interactive()

    if not goal or not goal.strip():
        print("[error] Empty instruction. Provide a goal argument or use --interactive and type something before Ctrl-D.")
        sys.exit(1)

    plan = llm_plan(args.model, goal, root, images)
    explanation = plan.get("explanation", "")
    edits = plan.get("edits", [])
    commands = plan.get("commands", [])

    if explanation:
        print(f"\n[Model plan]\n{explanation}\n")

    if edits:
        apply_edits(edits, root, args.unsafe)
    else:
        print("No edits proposed.")

    if commands:
        run_commands(commands, root, args.unsafe)
    else:
        print("No commands proposed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted by user.")
