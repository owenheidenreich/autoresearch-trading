"""Redirect the T6 validator's fixed output into this seat's private directory."""

from pathlib import Path


_ORIGINAL_READ_TEXT = Path.read_text
_ORIGINAL_WRITE_TEXT = Path.write_text
_REVIEW_ROOT = Path(
    "/Users/gduby/Documents/autoresearch-trading/"
    "v4/audit/autoresearch/"
    "protocol101_ft2_20_delta_scoped_review_attempt002/"
    "seat2_ml_statistics"
)
_VALIDATION_TARGET = Path(
    "/Users/gduby/Documents/autoresearch-trading/"
    "v4/audit/autoresearch/"
    "protocol101_ft2_08_data_tensor_label_contract/validation.json"
)
_VALIDATION_REDIRECT = _REVIEW_ROOT / "t6_ft208_validation.json"


def _redirected_read_text(path: Path, *args: object, **kwargs: object) -> str:
    if path.resolve() == _VALIDATION_TARGET and _VALIDATION_REDIRECT.is_file():
        return _ORIGINAL_READ_TEXT(_VALIDATION_REDIRECT, *args, **kwargs)
    return _ORIGINAL_READ_TEXT(path, *args, **kwargs)


def _redirected_write_text(
    path: Path, data: str, *args: object, **kwargs: object
) -> int:
    if path.resolve() == _VALIDATION_TARGET:
        return _ORIGINAL_WRITE_TEXT(
            _VALIDATION_REDIRECT, data, *args, **kwargs
        )
    return _ORIGINAL_WRITE_TEXT(path, data, *args, **kwargs)


Path.read_text = _redirected_read_text
Path.write_text = _redirected_write_text
