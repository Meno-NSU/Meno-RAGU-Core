from argparse import ArgumentParser
import codecs
import json
import os
import random


from dialogue_history_utils.dialogue_history_utils import prepare_dialogue_history


def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_fname', type=str, required=True,
                        help='The input JSON file with source dialogues in the chat template format.')
    parser.add_argument('-r', '--random', dest='random_seed', type=int, required=False,
                        default=None, help='The random seed.')
    parser.add_argument('--max_words', dest='max_words', type=int, required=False,
                        default=None, help='The maximum number of words in the assistant\'s response.')
    args = parser.parse_args()

    if args.random_seed is None:
        random_seed = random.randint(0, 2147483647)
    else:
        random_seed = args.random_seed
    print(f'The used random seed is {random_seed}.')
    random.seed(random_seed)

    input_fname = os.path.normpath(args.input_fname)
    if not os.path.isfile(input_fname):
        raise IOError(f'The file "{input_fname}" does not exist!')

    with codecs.open(input_fname, mode='r', encoding='utf-8') as fp:
        dialogue_examples = json.load(fp)
    dialogue_types = sorted(list(dialogue_examples.keys()))
    print(f'There are {len(dialogue_types)} dialogue types. They are: {dialogue_types}')
    selected_dialogue_type = random.choice(dialogue_types)
    info_msg = (f'Selected dialogue type is {selected_dialogue_type}. '
                f'There are {len(dialogue_examples[selected_dialogue_type])} dialogue examples of this type.')
    print(info_msg)
    selected_example = random.choice(dialogue_examples[selected_dialogue_type])['dialogue']
    print('\n')
    print(json.dumps(obj=selected_example, ensure_ascii=False, indent=4))
    print('\n')
    print(prepare_dialogue_history(source_dialogue=selected_example, max_words=args.max_words))


if __name__ == '__main__':
    main()
