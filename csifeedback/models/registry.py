"""模型注册表。

baseline 模型在各自模块中通过 ``@register_model`` 自注册。新增研究模型时，
只需在自己的模块里使用同一个装饰器，并在 ``csifeedback/models/__init__.py``
中导入该模块即可，无需修改任何已有模型。
"""

from typing import Any, Callable, Dict, List

from csifeedback.models.base import CSIAutoencoder


__all__ = [
    "register_model",
    "get_model",
    "get_model_from_config",
    "available_models",
    "_MODEL_FACTORIES",
]


ModelFactory = Callable[..., CSIAutoencoder]

_MODEL_FACTORIES: Dict[str, ModelFactory] = {}


def register_model(name: str) -> Callable[[ModelFactory], ModelFactory]:
    """把模型工厂注册到全局表。

    用法::

        @register_model("mynet")
        def create_mynet(encoded_dim: int = 512, **kwargs) -> CSIAutoencoder:
            return MyNet(encoded_dim=encoded_dim, **kwargs)
    """
    key = name.lower().strip()
    if not key:
        raise ValueError("模型名称不能为空")

    def decorator(factory: ModelFactory) -> ModelFactory:
        existing = _MODEL_FACTORIES.get(key)
        if existing is not None and existing is not factory:
            raise ValueError(
                f"模型 {key!r} 已由 {existing.__name__} 注册；"
                "请使用不同的名称或显式覆盖注册表。"
            )
        _MODEL_FACTORIES[key] = factory
        return factory

    return decorator


def get_model(name: str, **kwargs: Any) -> CSIAutoencoder:
    """按名称创建模型。"""
    key = name.lower().strip()
    factory = _MODEL_FACTORIES.get(key)
    if factory is None:
        raise ValueError(
            f"未知模型：{key}。可选：{available_models()}。"
            "新模型请在 csifeedback/models/__init__.py 中导入并注册。"
        )
    return factory(**kwargs)


def get_model_from_config(model_config: Any) -> CSIAutoencoder:
    """从 ``ModelConfig`` 创建模型，支持 ``extra`` 自定义参数。"""
    name = model_config.name.lower().strip()
    factory = _MODEL_FACTORIES.get(name)
    if factory is None:
        raise ValueError(
            f"未知模型：{name}。请在 csifeedback/models/__init__.py 中导入并注册"
            "对应的模型模块（使用 @register_model 装饰器）。"
        )

    kwargs: Dict[str, Any] = {}
    if model_config.reduction is not None:
        kwargs["reduction"] = model_config.reduction
    if model_config.encoded_dim is not None:
        kwargs["encoded_dim"] = model_config.encoded_dim
    kwargs.update(model_config.extra)
    return factory(**kwargs)


def available_models() -> List[str]:
    """返回已注册的模型名称。"""
    return sorted(_MODEL_FACTORIES)
