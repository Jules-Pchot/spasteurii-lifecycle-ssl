"""Claude Code hook: archive the session transcript (JSONL) as Markdown in the Obsidian vault.

Called on SessionEnd and PreCompact with the hook JSON on stdin (it holds "transcript_path" and "session_id").
Writes Transcripts/<date>-<session id>.md in the vault; the file is rewritten in full on every call, so the last
call of a session leaves the complete conversation. Only the standard library is used.

Manual run: python3 archive_transcript.py <transcript.jsonl>
"""

import json
import re
import sys
from pathlib import Path

VAULT_DIR = Path("/Users/juwul/Documents/Obsidian/Claude Obsidian/BacteriaSSL/Transcripts")
TOOL_INPUT_MAX = 600       # characters of a tool call input kept in the note
TOOL_RESULT_MAX = 1500     # characters of a tool result kept in the note

SYSTEM_TAGS = re.compile(r"<(system-reminder|ide_opened_file|ide_selection|local-command-caveat|command-[a-z-]+)>.*?</\1>", re.S)


def shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n… [{len(text) - limit} characters cut]"


def block_text(content) -> str:
    """Text of a tool_result content (string or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", f"[{b.get('type')}]") for b in content if isinstance(b, dict))
    return ""


def quote(text: str) -> str:
    return "\n".join("> " + line for line in text.splitlines())


def render(entries: list[dict]) -> str:
    out = []
    for e in entries:
        if e.get("isMeta") or e.get("isSidechain") or e.get("type") not in ("user", "assistant"):
            continue
        content = (e.get("message") or {}).get("content")
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else (content or [])
        for b in blocks:
            kind = b.get("type")
            if kind == "text":
                text = SYSTEM_TAGS.sub("", b.get("text", "")).strip()
                if text:
                    out.append(f"### {'User' if e['type'] == 'user' else 'Claude'}\n\n{text}\n")
            elif kind == "tool_use":
                args = shorten(json.dumps(b.get("input", {}), ensure_ascii=False, indent=1), TOOL_INPUT_MAX)
                out.append(f"> [!example]- Tool: {b.get('name')}\n{quote(args)}\n")
            elif kind == "tool_result":
                result = shorten(block_text(b.get("content")).strip(), TOOL_RESULT_MAX)
                if result:
                    out.append(f"> [!quote]- Result\n{quote(result)}\n")
    return "\n".join(out)


def main() -> None:
    if len(sys.argv) > 1:
        transcript, session, event = Path(sys.argv[1]), Path(sys.argv[1]).stem, "manual"
    else:
        payload = json.load(sys.stdin)
        transcript = Path(payload["transcript_path"]).expanduser()
        session, event = payload.get("session_id", transcript.stem), payload.get("hook_event_name", "")

    entries = []
    for line in transcript.read_text(encoding="utf-8").splitlines():
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    stamps = [e["timestamp"] for e in entries if e.get("timestamp")]
    date = stamps[0][:10] if stamps else "unknown-date"
    title = next((e.get("aiTitle") for e in reversed(entries) if e.get("aiTitle")), "Claude Code session")

    VAULT_DIR.mkdir(parents=True, exist_ok=True)
    note = VAULT_DIR / f"{date}-{session[:8]}.md"
    header = (f"---\ntitle: \"{title}\"\ndate: {date}\nsession_id: {session}\nlast_event: {event}\n"
              f"tags:\n  - transcript\n---\n\n# {title}\n\nRaw archive of the conversation (thinking omitted, "
              f"tool calls and results shortened). Summary notes are in [[Index]].\n\n")
    tmp = note.with_suffix(".tmp")
    tmp.write_text(header + render(entries), encoding="utf-8")
    tmp.replace(note)


if __name__ == "__main__":
    main()
