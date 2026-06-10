"""
CallLogger — structured per-call log file for debugging & tracking.

Log format (one line per event):
    [HH:MM:SS] #SEQ  LEVEL  STAGE  message

LEVEL codes:
    👤 USER    — caller utterance (after STT)
    🤖 AGENT   — agent reply (before TTS)
    📝 NOTE    — data captured by add_note tool
    🔍 RAG     — RAG retrieval pipeline step
    ⚙️  SYS     — lifecycle / system events
    ❌ ERR     — errors / exceptions

STAGE codes tell you exactly where in the pipeline the event happened:
    STT       — speech-to-text transcription
    LLM       — language model processing
    TTS       — text-to-speech synthesis
    RAG       — knowledge-base retrieval
    TOOL      — function_tool call
    LIFECYCLE — call start / end / summary
    ERROR     — error in any stage

This lets you reconstruct the exact sequence of steps when something goes wrong.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("call_logger")


class CallLogger:
    """Structured per-call log for debugging the full STT→RAG→LLM→TTS pipeline."""

    def __init__(self, room_name: str, logs_dir: str = "call_logs"):
        self.room_name = room_name
        self.start_time = datetime.now()
        self.logs_dir = logs_dir
        self._seq = 0  # monotonic sequence number across all events
        self._closed = False

        Path(self.logs_dir).mkdir(parents=True, exist_ok=True)

        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S_%f")
        safe_room_name = self._safe_filename(room_name)
        self.log_file = os.path.join(self.logs_dir, f"{safe_room_name}_{timestamp}.txt")

        # Open the file ONCE and keep the handle alive for the whole call.
        # This is the fix for the repeated-header bug: previously _append_to_file
        # did open(..., "a") on every write, so every new CallLogger instance
        # (re-initialized by LiveKit) would re-write the header into the same file.
        self._fh = open(self.log_file, "a", encoding="utf-8", buffering=1)  # line-buffered

        self._write_header()
        logger.info("✅ CallLogger ready → %s", self.log_file)

    # ──────────────────────────────────────────────────────────────────────────
    # Public logging methods
    # ──────────────────────────────────────────────────────────────────────────

    def log_user_message(self, message: str):
        """Log a caller utterance (output of STT)."""
        self._write("👤 USER ", "STT    ", message)

    def log_agent_message(self, message: str):
        """Log an agent reply (input to TTS)."""
        self._write("🤖 AGENT", "LLM    ", message)

    def log_note(self, note: str):
        """Log a note captured by the add_note tool."""
        self._write("📝 NOTE ", "TOOL   ", note)

    def log_system_event(self, event: str, stage: str = "LIFECYCLE"):
        """Log a lifecycle or pipeline step event."""
        self._write("⚙️  SYS  ", f"{stage:<7}", event)

    def log_rag_event(self, event: str):
        """Log a RAG-specific step (query, hit count, chars injected, etc.)."""
        self._write("🔍 RAG  ", "RAG    ", event)

    def log_error(self, error: str, stage: str = "ERROR"):
        """Log an error — include stage name so you know where it broke."""
        self._write("❌ ERR  ", f"{stage:<7}", error)
        logger.error("[%s] %s", stage, error)

    def log_call_summary(self, notes: list[str], duration_seconds: Optional[float] = None):
        """Write the end-of-call summary block."""
        if self._closed:
            return
        end_time = datetime.now()
        duration = duration_seconds or (end_time - self.start_time).total_seconds()
        mins, secs = divmod(int(duration), 60)

        lines = [
            "",
            "=" * 80,
            "📊 CALL SUMMARY",
            "=" * 80,
            f"End Time : {end_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Duration : {mins}m {secs}s",
            f"Events   : {self._seq} total",
            f"Notes    : {len(notes)}",
            "",
            "NOTES CAPTURED:",
        ]
        if notes:
            for i, note in enumerate(notes, 1):
                lines.append(f"  {i}. {note}")
        else:
            lines.append("  (none)")
        lines.append("=" * 80)

        self._append_to_file("\n".join(lines) + "\n")
        self._closed = True
        self._fh.flush()
        self._fh.close()
        logger.info("Call log saved → %s", self.log_file)

    def get_log_file_path(self) -> str:
        return self.log_file

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _write(self, level: str, stage: str, message: str):
        """Format and append one log line."""
        if self._closed:
            return
        ts = datetime.now().strftime("%H:%M:%S")
        seq = self._next_seq()
        line = f"[{ts}] #{seq:04d}  {level}  {stage}  {message}\n"
        self._append_to_file(line)

    def _write_header(self):
        # NOTE: explicit + between every segment avoids the Python implicit
        # string-literal-concat trap: "=" * 80 + "\n" f"X\n" "=" * 80
        # was parsed as ("=" * 80) + ("\nX\n=") * 80 — repeating 80 times.
        sep = "=" * 80
        ts  = self.start_time.strftime("%Y-%m-%d %H:%M:%S")
        header = (
            sep + "\n"
            + f"📞 CALL LOG  —  {self.room_name}\n"
            + sep + "\n"
            + f"Start Time : {ts}\n"
            + f"Room       : {self.room_name}\n"
            + "\n"
            + "COLUMNS: [time]  #seq  level  stage  message\n"
            + "STAGES : STT → RAG → LLM → TTS  |  TOOL  |  LIFECYCLE  |  ERROR\n"
            + sep + "\n\n"
        )
        self._append_to_file(header)

    def _append_to_file(self, content: str):
        try:
            self._fh.write(content)
        except Exception as exc:
            logger.error("Failed to write log: %s", exc)

    def __del__(self):
        """Safety net: close the handle if log_call_summary was never called."""
        try:
            if hasattr(self, "_fh") and not self._fh.closed:
                self._fh.flush()
                self._fh.close()
        except Exception:
            pass

    @staticmethod
    def _safe_filename(value: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
        return safe.strip("_") or "call"