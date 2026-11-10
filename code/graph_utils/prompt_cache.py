"""
Compile each RAGU prompt template once.

`ragu.common.prompts.messages.render` calls `_JINJA_ENV.from_string(content)` for every message of every row of
a batch, so the same template is compiled again for every chunk (3.5 ms per prompt in the graph build). A compiled
jinja2 template renders the same text however many times it is used, so caching it changes no prompt and no
LLM-cache key.
"""
import functools


def cache_prompt_templates() -> None:
    from ragu.common.prompts import messages

    env = messages._JINJA_ENV
    if getattr(env.from_string, '_compiled_once', False):
        return
    original = env.from_string
    compiled = functools.lru_cache(maxsize=4096)(lambda source: original(source))

    def from_string(source, globals=None, template_class=None):
        if globals is None and template_class is None and isinstance(source, str):
            return compiled(source)
        return original(source, globals, template_class)

    from_string._compiled_once = True
    env.from_string = from_string
