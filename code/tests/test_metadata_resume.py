import json
from collections import Counter

import httpx
import openai
import pytest

import generate_title_and_annotation as metadata


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')


def enriched(document, *, title='Сохранённое название', annotation='Сохранённая аннотация.'):
    row = dict(document)
    row['doc_full_text'] = row.pop('text')
    return dict(row, doc_title=title, doc_annotation=annotation)


@pytest.fixture
def run_metadata(tmp_path, monkeypatch):
    documents = [
        {'text': f'Документ номер {idx}.', 'url': f'https://nsu.ru/{idx}',
         'chunks': [{'start_char': 0, 'end_char': 17}], 'content_source': 'base'}
        for idx in range(12)
    ]
    source = tmp_path / 'input.jsonl'
    output = tmp_path / 'metadata.jsonl'
    write_jsonl(source, documents)
    calls = Counter()
    replies = {}

    def respond(request):
        prompt = json.loads(request.content)['messages'][-1]['content']
        idx = next(i for i, doc in enumerate(documents) if doc['text'] in prompt)
        calls[idx] += 1
        content = replies.get(idx, [json.dumps({'title': 'Новое название', 'annotation': 'Новая аннотация.'})])
        content = content[min(calls[idx] - 1, len(content) - 1)]
        return httpx.Response(200, json={
            'id': 'test', 'object': 'chat.completion', 'created': 1, 'model': 'test',
            'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': content}}],
        })

    client = httpx.Client(transport=httpx.MockTransport(respond))
    original = openai.OpenAI
    monkeypatch.setattr(metadata.openai, 'OpenAI', lambda **kwargs: original(**kwargs, http_client=client))
    monkeypatch.setattr(metadata.time, 'sleep', lambda delay: None)

    def run(retries=0):
        monkeypatch.setattr(metadata.sys, 'argv', [
            'metadata', '-i', str(source), '-o', str(output), '-m', 'test', '--url', 'http://llm.test/v1',
            '--resume', '--fail_on_error', '--max_retries', str(retries), '--concurrency', '1',
        ])
        metadata.main()
        return [json.loads(line) for line in output.read_text().splitlines()]

    yield run, documents, source, output, calls, replies
    client.close()


def test_resume_keeps_successes_after_five_scattered_errors(run_metadata):
    run, documents, _, output, calls, _ = run_metadata
    failed = {0, 2, 5, 8, 10}
    original = [enriched(doc, title='' if idx in failed else f'Название {idx}') for idx, doc in enumerate(documents)]
    write_jsonl(output, original)
    result = run()
    assert calls == Counter({idx: 1 for idx in failed})
    assert len(result) == len(documents)
    for idx in range(len(documents)):
        assert result[idx] == (enriched(documents[idx], title='Новое название', annotation='Новая аннотация.')
                               if idx in failed else original[idx])
    assert not output.with_suffix('.jsonl.progress.jsonl').exists()


@pytest.mark.parametrize('invalid', ['invalid JSON', None, '{"title":"", "annotation":"текст"}'])
def test_invalid_metadata_is_retried_automatically(run_metadata, invalid):
    run, documents, _, output, calls, replies = run_metadata
    write_jsonl(output, [enriched(doc) for doc in documents[0:11]])
    replies[11] = [invalid, '{"title":"Название", "annotation":"Аннотация."}']
    result = run(retries=1)
    assert calls == Counter({11: 2})
    assert result[11]['doc_title'] == 'Название'


def test_permanent_failures_preserve_successes_for_next_run(run_metadata):
    run, documents, _, output, calls, replies = run_metadata
    replies[2] = ['bad JSON']
    with pytest.raises(RuntimeError, match='failed for 1 documents'):
        run(retries=1)
    saved = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(saved) == len(documents)
    assert saved[2]['doc_title'] == ''
    calls.clear()
    replies.clear()
    result = run()
    assert calls == Counter({2: 1})
    assert all(result[i] == saved[i] for i in range(len(documents)) if i != 2)


def test_resume_recovers_records_written_before_interruption(run_metadata, monkeypatch):
    run, documents, _, output, calls, _ = run_metadata
    original = metadata.as_completed

    def interrupted(futures):
        yield next(original(futures))
        raise KeyboardInterrupt

    with monkeypatch.context() as context:
        context.setattr(metadata, 'as_completed', interrupted)
        with pytest.raises(KeyboardInterrupt):
            run()
    progress = output.with_suffix('.jsonl.progress.jsonl')
    saved = [json.loads(line) for line in progress.read_text().splitlines() if line.strip()]
    assert len(saved) == 1
    saved_idx = saved[0]['doc_idx']
    with progress.open('a') as stream:
        stream.write('{"doc_idx":')
    calls.clear()
    result = run()
    assert len(result) == len(documents)
    assert calls == Counter({i: 1 for i in range(len(documents)) if i != saved_idx})


@pytest.mark.parametrize('change', ['text', 'chunks', 'url'])
def test_resume_rejects_changed_input_without_modifying_saved_results(run_metadata, change):
    run, documents, source, output, calls, _ = run_metadata
    write_jsonl(output, [enriched(doc) for doc in documents])
    before = output.read_bytes()
    changed = [dict(doc) for doc in documents]
    changed[1][change] = [{'start_char': 0, 'end_char': 10}] if change == 'chunks' else 'different'
    write_jsonl(source, changed)
    with pytest.raises(ValueError, match='не соответствуют'):
        run()
    assert not calls
    assert output.read_bytes() == before


def test_atomic_publish_failure_keeps_all_completed_responses(run_metadata, monkeypatch):
    run, documents, _, output, calls, _ = run_metadata
    original = [enriched(doc, title='' if idx == 0 else 'Сохранённое название') for idx, doc in enumerate(documents)]
    write_jsonl(output, original)
    before = output.read_bytes()
    with monkeypatch.context() as context:
        context.setattr(metadata.os, 'replace', lambda *args: (_ for _ in ()).throw(OSError('publish failed')))
        with pytest.raises(OSError, match='publish failed'):
            run()
    assert output.read_bytes() == before
    calls.clear()
    result = run()
    assert not calls
    assert result[0]['doc_title'] == 'Новое название'
    assert result[1:] == original[1:]


def test_malformed_output_record_does_not_discard_later_successes(run_metadata):
    run, documents, _, output, calls, _ = run_metadata
    lines = [json.dumps(enriched(doc)) for doc in documents]
    lines[1] = '{broken'
    output.write_text('\n'.join(lines) + '\n')
    assert len(run()) == len(documents)
    assert calls == Counter({1: 1})
