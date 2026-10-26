from argparse import ArgumentParser
import codecs
import json
import os
from typing import Dict, List, Union


def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The input JSON file with full information about NSU teachers and employees.')
    parser.add_argument('-c', '--chunks', dest='chunks_name', type=str, required=True,
                        help='The input JSON file with pre-rendered markdown chunks of the teacher profiles. '
                             'Its records must be aligned with the records of the input JSON file by indices.')
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='The output JSONL file with teacher documents for the combine_data.py script.')
    parser.add_argument('--include_empty_schedule', dest='include_empty_schedule', action='store_true',
                        help='Will the teachers without any lessons in their timetables be exported? '
                             'By default, only the teachers having at least one lesson are exported.')
    args = parser.parse_args()

    input_fname = os.path.normpath(args.input_name)
    if not os.path.isfile(input_fname):
        raise IOError(f'The file "{input_fname}" does not exist!')

    chunks_fname = os.path.normpath(args.chunks_name)
    if not os.path.isfile(chunks_fname):
        raise IOError(f'The file "{chunks_fname}" does not exist!')

    output_fname = os.path.normpath(args.output_name)
    if not os.path.isfile(output_fname):
        output_dirname = os.path.dirname(output_fname)
        if len(output_dirname) > 0:
            if not os.path.isdir(output_dirname):
                raise IOError(f'The directory "{output_dirname}" does not exist!')

    with codecs.open(input_fname, mode='r', encoding='utf-8') as src_fp:
        teachers: List[Dict[str, Union[str, List[Dict[str, Union[str, int, List[str]]]]]]] = json.load(src_fp)
    with codecs.open(chunks_fname, mode='r', encoding='utf-8') as src_fp:
        chunked_teachers: List[Dict[str, List[str]]] = json.load(src_fp)

    if len(teachers) != len(chunked_teachers):
        err_msg = (f'The files "{input_fname}" and "{chunks_fname}" are not aligned! '
                   f'{len(teachers)} != {len(chunked_teachers)}.')
        raise IOError(err_msg)

    num_exported_teachers = 0
    num_exported_chunks = 0
    num_skipped_teachers = 0
    with codecs.open(output_fname, mode='w', encoding='utf-8', buffering=0) as dst_fp:
        for teacher_idx, (cur_teacher, cur_chunked_teacher) in enumerate(zip(teachers, chunked_teachers)):
            teacher_name = cur_teacher['name'].strip()
            teacher_chunks = cur_chunked_teacher.get('chunks', [])
            if len(teacher_chunks) == 0:
                err_msg = (f'The teacher "{teacher_name}" (the record {teacher_idx}) does not contain '
                           f'any text chunks!')
                raise IOError(err_msg)
            first_chunk_line = teacher_chunks[0].split('\n', 1)[0].lstrip('#').strip()
            if first_chunk_line != teacher_name:
                err_msg = (f'The files "{input_fname}" and "{chunks_fname}" are not aligned! '
                           f'The record {teacher_idx} of the first file is "{teacher_name}", '
                           f'but the record of the second file is "{first_chunk_line}".')
                raise IOError(err_msg)
            if (not args.include_empty_schedule) and (len(cur_teacher.get('schedule', [])) == 0):
                num_skipped_teachers += 1
                continue
            schedule_url = cur_teacher.get('schedule_url', '').strip()
            for cur_chunk in teacher_chunks:
                prepared_text = cur_chunk.strip()
                if len(prepared_text) == 0:
                    continue
                new_sample = {
                    'text': prepared_text,
                    'url': schedule_url
                }
                dst_fp.write(json.dumps(obj=new_sample, ensure_ascii=False) + '\n')
                num_exported_chunks += 1
            num_exported_teachers += 1

    print(f'{num_exported_teachers} teachers ({num_exported_chunks} self-contained text chunks) are '
          f'successfully saved into "{output_fname}".')
    if num_skipped_teachers > 0:
        print(f'{num_skipped_teachers} teachers without lessons in their timetables are skipped.')


if __name__ == '__main__':
    main()
