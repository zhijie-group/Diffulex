from __future__ import annotations

from typing import Any, Callable

from diffulex.config import Config
from diffulex.utils.loader import load_model


_NOT_PROVIDED = object()
RegistryEntry = tuple[Callable[[Any], Any] | type | None, bool]

class AutoModelForDiffusionLM:
    """Factory and registry for diffusion language models."""

    MODEL_MAPPING: dict[str, RegistryEntry] = {}

    @classmethod
    def register(
        cls,
        model_name: str,
        model_class: Callable[[Any], Any] | type | None = _NOT_PROVIDED,
        *,
        use_full_config: bool = False,
        exist_ok: bool = False,
    ):
        """Register a model factory or class under ``model_name``.

        When ``model_class`` is omitted this method returns a decorator.

        Args:
            model_name: Key used to retrieve the model.
            model_class: Callable or class that builds the model instance.
            use_full_config: Pass the entire :class:`Config` to the factory
                instead of ``config.hf_config``.
            exist_ok: Allow overriding an existing registration.
        """

        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model_name must be a non-empty string.")

        if model_class is _NOT_PROVIDED:
            def decorator(model_cls):
                cls._register(model_name, model_cls, use_full_config=use_full_config, exist_ok=exist_ok)
                return model_cls

            return decorator

        cls._register(model_name, model_class, use_full_config=use_full_config, exist_ok=exist_ok)
        return model_class

    @classmethod
    def _register(
        cls,
        model_name: str,
        model_class: Callable[[Any], Any] | type | None,
        *,
        use_full_config: bool,
        exist_ok: bool,
    ) -> None:
        if not exist_ok and model_name in cls.MODEL_MAPPING:
            raise ValueError(f"Model '{model_name}' is already registered.")
        cls.MODEL_MAPPING[model_name] = (model_class, use_full_config)

    @classmethod
    def unregister(cls, model_name: str) -> None:
        cls.MODEL_MAPPING.pop(model_name, None)

    @classmethod
    def available_models(cls) -> tuple[str, ...]:
        return tuple(sorted(cls.MODEL_MAPPING))

    @classmethod
    def from_config(cls, config: Config):
        if not hasattr(config, "model_name"):
            raise AttributeError("Config must define 'model_name' to build a model.")

        try:
            factory, use_full_config = cls.MODEL_MAPPING[config.model_name]
        except KeyError as err:
            available = ", ".join(cls.available_models()) or "<none>"
            raise ValueError(
                f"Model '{config.model_name}' is not registered. Available models: {available}."
            ) from err

        if factory is None:
            raise ValueError(f"Model '{config.model_name}' is reserved but not implemented yet.")

        init_arg = config if use_full_config else config.hf_config
        if init_arg is None:
            raise ValueError("Config.hf_config must be initialized before building the model.")

        model = factory(init_arg)
        return load_model(model, config)

# Backwards compatibility with the old name while callers migrate.
AutoModelLM = AutoModelForDiffusionLM