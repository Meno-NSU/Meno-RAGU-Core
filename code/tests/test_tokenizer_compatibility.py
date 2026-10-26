import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from tokenization_utils.tokenization_utils import load_hf_tokenizer


@pytest.mark.parametrize('extra', [['<|video|>'], {'image_token': '<|image|>'}])
def test_new_and_legacy_special_tokens_load_without_changing_ids_or_files(tmp_path, extra):
    vocab = {'<unk>': 0, '<bos>': 1, '<eos>': 2, '<pad>': 3, '<|video|>': 4,
             '<|image|>': 5, 'Привет': 6, 'НГУ': 7}
    backend = Tokenizer(models.WordLevel(vocab, unk_token='<unk>'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    template = "{{ bos_token }}{% for message in messages %}{{ message['content'] }}{% endfor %}"
    PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='<unk>', bos_token='<bos>',
                           eos_token='<eos>', pad_token='<pad>', chat_template=template).save_pretrained(tmp_path)
    config_path = tmp_path / 'tokenizer_config.json'
    config = json.loads(config_path.read_text())
    config['tokenizer_class'] = 'GemmaTokenizer'
    config['extra_special_tokens'] = extra
    config['processor_class'] = 'Gemma4UnifiedProcessor'
    config_path.write_text(json.dumps(config))
    (tmp_path / 'config.json').write_text('{"model_type": "gemma4_unified"}')
    before = config_path.read_bytes()
    if isinstance(extra, list):
        with pytest.raises(AttributeError, match='keys'):
            AutoTokenizer.from_pretrained(tmp_path)
    loaded = load_hf_tokenizer(str(tmp_path))
    assert loaded.convert_tokens_to_ids('<|video|>') == 4
    assert loaded.convert_tokens_to_ids('<|image|>') == 5
    assert loaded.tokenize('Привет НГУ') == ['Привет', 'НГУ']
    assert loaded.apply_chat_template([{'role': 'user', 'content': 'Привет НГУ'}],
                                      tokenize=True, add_generation_prompt=True) == [1, 6, 7]
    assert loaded.chat_template == template
    assert loaded.get_vocab() == vocab
    assert config_path.read_bytes() == before
