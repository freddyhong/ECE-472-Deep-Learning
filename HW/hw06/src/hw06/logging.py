import logging
import os
import sys
from pathlib import Path

import jax
import numpy as np
import structlog


class FormattedFloat(float):
    def __repr__(self) -> str:
        return f"{self:.4g}"


def custom_serializer_processor(logger, method_name, event_dict):
    for key, value in event_dict.items():
        if hasattr(value, "numpy"): 
            value = value.numpy()
        if isinstance(value, jax.Array):
            value = np.array(value)
        if isinstance(value, (np.generic, np.ndarray)):
            value = value.item() if value.size == 1 else value.tolist()
        if isinstance(value, float):
            value = FormattedFloat(value)
        if isinstance(value, Path):
            value = str(value)
        event_dict[key] = value
    return event_dict

import logging
import sys
from pathlib import Path
import structlog

# keep your existing helpers if you have them:
# - FormattedFloat
# - custom_serializer_processor

def configure_logging(
    log_dir: Path = Path("hw06/artifacts"),
    log_name_json: str = "log.json",
    log_name_txt: str = "log.txt",
):
    log_dir.mkdir(parents=True, exist_ok=True)
    json_path = log_dir / log_name_json
    txt_path  = log_dir / log_name_txt

    shared_processors = [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        custom_serializer_processor,  
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,  
    ]

    structlog.configure(
        processors=shared_processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    console_renderer = structlog.dev.ConsoleRenderer(
        colors=sys.stdout.isatty(),
        exception_formatter=structlog.dev.RichTracebackFormatter(),
    )
    json_renderer = structlog.processors.JSONRenderer()
    text_renderer = structlog.dev.ConsoleRenderer(colors=False)  

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                console_renderer,
            ]
        )
    )

    json_file_handler = logging.FileHandler(json_path, mode="w", encoding="utf-8")
    json_file_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                json_renderer,
            ]
        )
    )

    text_file_handler = logging.FileHandler(txt_path, mode="w", encoding="utf-8")
    text_file_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                text_renderer,
            ]
        )
    )

    root = logging.getLogger()
    root.handlers.clear() 
    root.addHandler(console_handler)
    root.addHandler(json_file_handler)
    root.addHandler(text_file_handler)

    log_level = logging.getLevelName((sys.argv and "INFO") or "INFO")
    root.setLevel(log_level)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    return {"json": json_path, "txt": txt_path}
