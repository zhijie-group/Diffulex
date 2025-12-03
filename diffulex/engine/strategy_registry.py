from typing import Iterable

from diffulex.utils.registry import fetch_factory_name


_NOT_PROVIDED = object()


class DiffulexStrategyRegistry:
    """Registry-driven factory for module implementations."""
    
    _DEFAULT_KEY = "__default__"
    
    def __init_subclass__(cls, **kwargs):
        """Initialize a separate _MODULE_MAPPING for each subclass."""
        super().__init_subclass__(**kwargs)
        cls._MODULE_MAPPING: dict[str, object] = {} 
    
    @classmethod
    def register(
        cls,
        strategy_name: str,
        factory: object = _NOT_PROVIDED,
        *,
        aliases: Iterable[str] = (),
        is_default: bool = False,
        exist_ok: bool = False,
    ):
        if not isinstance(strategy_name, str) or not strategy_name:
            raise ValueError("strategy_name must be a non-empty string.")
        if isinstance(aliases, str):
            raise TypeError("aliases must be an iterable of strings, not a single string.")

        def decorator(factory_fn: object):
            cls._register(strategy_name, factory_fn, exist_ok=exist_ok)
            for alias in dict.fromkeys(aliases):
                if not isinstance(alias, str) or not alias:
                    raise ValueError("aliases must contain non-empty strings.")
                cls._register(alias, factory_fn, exist_ok=exist_ok)
            if is_default:
                cls._register(cls._DEFAULT_KEY, factory_fn, exist_ok=True)
            return factory_fn

        if factory is _NOT_PROVIDED:
            return decorator
        return decorator(factory)

    @classmethod
    def _register(cls, key: str, factory: object, *, exist_ok: bool) -> None:
        # If the same factory is already registered, silently skip (idempotent registration)
        if key in cls._MODULE_MAPPING:
            existing = cls._MODULE_MAPPING[key]
            # Check if it's the same factory object
            if existing is factory:
                return  # Same factory already registered, no-op
            # Check if it's the same class by name and module (handles module reload cases)
            existing_name = fetch_factory_name(existing)
            new_name = fetch_factory_name(factory)
            if existing_name == new_name:
                return  # Same class already registered (possibly from module reload), no-op
            if not exist_ok:
                raise ValueError(
                    f"Module '{key}: {new_name}' is already registered as '{existing_name}'. "
                    f"Use exist_ok=True to override."
                )
        cls._MODULE_MAPPING[key] = factory
    
    @classmethod
    def unregister(cls, strategy_name: str) -> None:
        cls._MODULE_MAPPING.pop(strategy_name, None)
        
    @classmethod
    def available_modules(cls) -> tuple[str, ...]:
        return tuple(sorted(k for k in cls._MODULE_MAPPING if k != cls._DEFAULT_KEY))