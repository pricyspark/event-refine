from pathlib import Path


def resolve_pretrained_source(model_name_or_path: str) -> tuple[str, bool]:
    """Return a local source unchanged and let Hugging Face resolve remote caches."""
    path = Path(model_name_or_path).expanduser()
    if path.exists():
        return str(path.resolve()), True
    normalized = model_name_or_path.replace("\\", "/")
    looks_local = (
        path.is_absolute()
        or normalized.startswith(("./", "../", "~/"))
        or normalized.count("/") > 1
    )
    if looks_local:
        raise FileNotFoundError(
            f"Local pretrained model path does not exist: {path}. "
            "Pass an existing checkpoint directory or a Hugging Face model ID."
        )
    return model_name_or_path, False
