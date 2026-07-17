"""Simulation components used by the parameter-identification runtime."""

# Avoid circular imports by using lazy imports.
__all__ = [
    'CheckpointLoader',
    'EnvironmentFactory',
    'CognitiveModuleManager',
    'ActionProcessor',
]

__version__ = "2.0.0"


def __getattr__(name):
    """Lazy-import module members to avoid circular dependencies."""
    if name == 'CheckpointLoader':
        from .checkpoint_loader import CheckpointLoader
        return CheckpointLoader
    elif name == 'EnvironmentFactory':
        from .environment_factory import EnvironmentFactory
        return EnvironmentFactory
    elif name == 'CognitiveModuleManager':
        from .cognitive_module_manager import CognitiveModuleManager
        return CognitiveModuleManager
    elif name == 'ActionProcessor':
        from .action_processor import ActionProcessor
        return ActionProcessor
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
