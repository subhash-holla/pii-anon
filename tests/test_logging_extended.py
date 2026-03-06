import logging

from pii_anon.observability.logging import JsonFormatter, get_logger


def test_json_formatter_with_payload() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord("x", logging.INFO, "", 0, "hello", (), None)
    setattr(record, "payload", {"a": 1})
    formatted = formatter.format(record)
    assert '"message": "hello"' in formatted
    assert '"a": 1' in formatted


def test_get_logger_reuses_existing_handlers() -> None:
    logger = get_logger("pii.core.test", level="INFO", structured=False)
    assert logger.handlers
    same = get_logger("pii.core.test", level="DEBUG", structured=True)
    assert same is logger
