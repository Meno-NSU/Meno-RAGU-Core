from ragu.common.prompts import messages

from graph_utils.prompt_cache import cache_prompt_templates


def test_template_is_compiled_once_and_renders_the_same(monkeypatch):
    env = messages._JINJA_ENV
    calls = []
    original = env.from_string

    def counting(source, *args, **kwargs):
        calls.append(source)
        return original(source, *args, **kwargs)

    monkeypatch.setattr(env, 'from_string', counting)
    cache_prompt_templates()
    first = env.from_string('Привет, {{ name }}!').render(name='НГУ')
    second = env.from_string('Привет, {{ name }}!').render(name='ММФ')
    assert (first, second) == ('Привет, НГУ!', 'Привет, ММФ!')
    assert calls == ['Привет, {{ name }}!']


def test_patching_twice_keeps_one_cache(monkeypatch):
    env = messages._JINJA_ENV
    monkeypatch.setattr(env, 'from_string', env.from_string)
    cache_prompt_templates()
    patched = env.from_string
    cache_prompt_templates()
    assert env.from_string is patched
