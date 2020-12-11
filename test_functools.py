import functools

@functools.lru_cache(500)
def calc(*args)->int:
    return " ".join(args)

print(calc("test"))
print(calc("test1", "test2"))
print(calc("test2", "test1"))
print(calc("test1", "test2"))
print(calc("test"))
print(calc.cache_info())
