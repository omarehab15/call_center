import logging
import os
from datetime import datetime
from typing import Optional, Any
from pathlib import Path

logger = logging.getLogger("call_logger")


class CallLogger:
    """Manages readable logging for each room/call session."""
    
    def __init__(self, room_name: str, logs_dir: str = "call_logs"):
        """
        Initialize CallLogger for a specific room.
        
        Args:
            room_name: The LiveKit room name
            logs_dir: Directory to store call logs (default: "call_logs")
        """
        self.room_name = room_name
        self.start_time = datetime.now()
        self.logs_dir = logs_dir
        
        # Create logs directory if it doesn't exist
        Path(self.logs_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created logs directory: {self.logs_dir}")
        
        # Create the log file path
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(
            self.logs_dir,
            f"{room_name}_{timestamp}.txt"
        )
        
        logger.info(f"Call log file path: {self.log_file}")
        
        # Initialize log file with header
        self._write_header()
        logger.info(f"✅ Call logger initialized for room: {room_name}")
    
    def _write_header(self):
        """Write header information to the log file."""
        header = f"""{'='*80}
📞 CALL LOG - {self.room_name}
{'='*80}
Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
Room Name: {self.room_name}
{'='*80}

"""
        self._append_to_file(header)
    
    def _append_to_file(self, content: str):
        """Append content to the log file."""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            logger.error(f"Failed to write to log file {self.log_file}: {e}")
    
    def log_user_message(self, message: str, timestamp: Optional[datetime] = None):
        """Log a user message."""
        ts = timestamp or datetime.now()
        time_str = ts.strftime("%H:%M:%S")
        entry = f"[{time_str}] 👤 USER: {message}\n"
        self._append_to_file(entry)
    
    def log_agent_message(self, message: str, timestamp: Optional[datetime] = None):
        """Log an agent message."""
        ts = timestamp or datetime.now()
        time_str = ts.strftime("%H:%M:%S")
        entry = f"[{time_str}] 🤖 AGENT: {message}\n"
        self._append_to_file(entry)
    
    def log_note(self, note: str, timestamp: Optional[datetime] = None):
        """Log a saved note."""
        ts = timestamp or datetime.now()
        time_str = ts.strftime("%H:%M:%S")
        entry = f"[{time_str}] 📝 NOTE: {note}\n"
        self._append_to_file(entry)
    
    def log_system_event(self, event: str, timestamp: Optional[datetime] = None):
        """Log a system event."""
        ts = timestamp or datetime.now()
        time_str = ts.strftime("%H:%M:%S")
        entry = f"[{time_str}] ⚙️  SYSTEM: {event}\n"
        self._append_to_file(entry)
    
    def log_error(self, error: str, timestamp: Optional[datetime] = None):
        """Log an error."""
        ts = timestamp or datetime.now()
        time_str = ts.strftime("%H:%M:%S")
        entry = f"[{time_str}] ❌ ERROR: {error}\n"
        self._append_to_file(entry)
    
    def log_call_summary(self, notes: list[str], duration_seconds: Optional[float] = None):
        """Log a call summary at the end of the call."""
        end_time = datetime.now()
        duration = duration_seconds or (end_time - self.start_time).total_seconds()
        duration_str = self._format_duration(duration)
        
        summary = f"""
{'='*80}
📊 CALL SUMMARY
{'='*80}
End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
Duration: {duration_str}
Total Notes: {len(notes)}

NOTES CAPTURED:
"""
        self._append_to_file(summary)
        
        if notes:
            for i, note in enumerate(notes, 1):
                self._append_to_file(f"  {i}. {note}\n")
        else:
            self._append_to_file("  (No notes captured)\n")
        
        footer = f"\n{'='*80}\n"
        self._append_to_file(footer)
        
        logger.info(f"Call log saved to {self.log_file}")
    
    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in seconds to a readable format."""
        minutes = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{minutes}m {secs}s"
    
    def get_log_file_path(self) -> str:
        """Get the full path to the log file."""
        return self.log_file
