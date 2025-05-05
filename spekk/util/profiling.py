import time
import cProfile, pstats, io
from enum import Enum, auto
import dataclasses
from spekk import ops

class Level(Enum):
    Error = auto()
    Warning = auto()
    Information = auto()
    Debug = auto()
    Silence = auto()

level = Level.Warning
accumulated_time = 0

def set_level(new_level: Level):
    global level
    level = new_level

def get_level():
    return level

def function_profiling(func=None, *, tag=None, function_level=Level.Warning):
    # If "level" > "function_profiling_leve"l" enabled the "function_profiling" decorator
    if func is None:
        def outer_wrapper(func):
            return function_profiling(func, tag=tag, function_level=function_level)
        return outer_wrapper
    
    else:
        def wrapper(*args, **kwargs):
            if level.value <= function_level.value:
                start_time = time.time()
                result = func(*args, **kwargs)
                call_block_until_ready(result)
                diff = time.time() - start_time
                header = f"[Profiling] {'' if tag is None else tag} {func.__name__}".ljust(35)
                global accumulated_time
                accumulated_time += diff
                print(f"{header} {diff:0.7f} seconds to execute. | accumulated_time={accumulated_time:0.4f}")
            else:
                result = func(*args, **kwargs)
            return result
            # return result
        return wrapper

def call_block_until_ready(result):
    "Force a single block_until_ready call to measure time"
    if ops.backend.backend_name != "jax":
        return result
    
    loop_over_attributes(result)

def loop_over_attributes(obj):

    if is_spekk_jax_array(obj):
        obj.data.block_until_ready()
        return

    if dataclasses.is_dataclass(obj):
        for _field in dataclasses.fields(obj):
            value = getattr(obj, _field.name)
            if is_spekk_jax_array(value):           
                value.data.block_until_ready()
        return
    
    if isinstance(obj, (list, tuple)):
        for value in obj:
            if is_spekk_jax_array(value):
                value.data.block_until_ready()
        return

    if isinstance(obj, dict):
        for key, value in obj.items():
            if is_spekk_jax_array(value):                
                value.data.block_until_ready()
        return


def is_jax_array(x):
    import jax.numpy as jnp
    return isinstance(x, jnp.ndarray)

def is_spekk_jax_array(obj):
    if isinstance(obj, ops.array):
        if is_jax_array(obj.data):
            return True
    return False

def full_cProfiling(fnc):
    
    """A decorator that uses cProfile to profile a function"""
    
    def inner(*args, **kwargs):
        
        profiler = cProfile.Profile()
        profiler.enable()
        retval = fnc(*args, **kwargs)
        profiler.disable()

        # Create a Stats object
        stats = pstats.Stats(profiler)
        # Sort by cumulative time and print the top 10 lines
        stats.sort_stats('cumulative').print_stats(50)

        return retval

    return inner
