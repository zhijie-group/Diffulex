from typing import Optional, Type

def check_config_diff(
    current_config: Optional[Type] = None,
    default_config_cls: Optional[Type] = None, 
    compare_to: str = "default",  # "default" or "existing"
):
    pass