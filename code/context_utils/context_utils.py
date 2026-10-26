import json
from typing import Any, Dict, List, Tuple, Union


def normalize_urls(value: Union[str, List[str], None]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    if isinstance(value, (list, tuple)):
        result: List[str] = []
        for v in value:
            s = v.strip() if isinstance(v, str) else str(v).strip()
            if s and s not in result:
                result.append(s)
        return result
    s = str(value).strip()
    return [s] if s else []


def build_reverse_chunk_index(documents: List[Dict[str, Union[str, float, List[Dict[str, int]]]]]
                              ) -> Dict[str, Dict[str, int]]:
    """
    Builds the mapping of global chunk indices to their local coordinates (the document
    index and the chunk index inside the document) over the documents in their final
    corpus order. The chunk order inside a document follows the order of boundaries in
    the 'chunks' field. An error is raised if at least one chunk has an empty text.
    """
    reverse_chunk_index: Dict[str, Dict[str, int]] = dict()
    global_chunk_idx = 0
    indices_of_empty_chunks: List[int] = []
    for document_index, cur_document in enumerate(documents):
        try:
            doc_text = cur_document['doc_full_text']
        except:
            doc_text = cur_document['text']
        for local_chunk_index, cur_chunk in enumerate(cur_document['chunks']):
            chunk_text = doc_text[cur_chunk['start_char']:cur_chunk['end_char']]
            if len(chunk_text.strip()) == 0:
                indices_of_empty_chunks.append(global_chunk_idx)
            reverse_chunk_index[str(global_chunk_idx)] = {
                'doc_index': document_index,
                'local_chunk_index': local_chunk_index
            }
            global_chunk_idx += 1
    if len(indices_of_empty_chunks) > 0:
        err_msg = f'The knowledge corpus contains empty texts of the following chunks: {indices_of_empty_chunks}'
        raise ValueError(err_msg)
    return reverse_chunk_index


def global_chunk_index_to_local(global_chunk_index: int,
                               documents: List[Dict[str, Union[str, float, List[Dict[str, int]]]]],
                               chunk_mapping: Dict[str, Dict[str, int]]) -> Tuple[int, int]:
    if str(global_chunk_index) not in chunk_mapping:
        err_msg = f'The global chunk index {global_chunk_index} is not supported in the chunk mapping!'
        raise ValueError(err_msg)
    document_index = chunk_mapping[str(global_chunk_index)]['doc_index']
    local_chunk_index = chunk_mapping[str(global_chunk_index)]['local_chunk_index']
    if (document_index < 0) or (document_index >= len(documents)):
        err_msg = f'The chunk mapping is wrong. The global chunk {global_chunk_index} contains a wrong data!\n'
        err_msg += json.dumps(chunk_mapping[str(global_chunk_index)], ensure_ascii=False, indent=4)
        raise ValueError(err_msg)
    if (local_chunk_index < 0) or (local_chunk_index >= len(documents[document_index]['chunks'])):
        err_msg = f'The chunk mapping is wrong. The global chunk {global_chunk_index} contains a wrong data!\n'
        err_msg += json.dumps(chunk_mapping[str(global_chunk_index)], ensure_ascii=False, indent=4)
        raise ValueError(err_msg)
    return document_index, local_chunk_index


def global_chunk_index_to_text(global_chunk_index: int,
                               documents: List[Dict[str, Union[str, float, List[Dict[str, int]]]]],
                               chunk_mapping: Dict[str, Dict[str, int]]) -> str:
    document_index, local_chunk_index = global_chunk_index_to_local(global_chunk_index, documents, chunk_mapping)
    try:
        doc_text = documents[document_index]['doc_full_text']
    except:
        doc_text = documents[document_index]['text']
    doc_chunks = documents[document_index]['chunks']
    chunk_boundaries = doc_chunks[local_chunk_index]
    chunk_text = doc_text[chunk_boundaries['start_char']:chunk_boundaries['end_char']]
    return chunk_text.strip()


def document_to_text(document_index: int, chunk_indices: List[int],
                     documents: List[Dict[str, Union[str, float, List[Dict[str, int]]]]]) -> str:
    doc_text = documents[document_index]['doc_full_text']
    doc_chunks = documents[document_index]['chunks']
    doc_title = ' '.join(documents[document_index]['doc_title'].strip().split()).strip()
    doc_annotation = ' '.join(documents[document_index]['doc_annotation'].strip().split()).strip()
    if len(doc_title) > 0:
        new_document_description = f'**Название документа:** {doc_title}'
        if doc_title[-1] not in {'.', ',', '?', '!', ':', ';'}:
            new_document_description += '.'
    else:
        new_document_description = 'Документ без названия.'
    if len(doc_annotation) > 0:
        new_document_description += f'\n\n**Аннотация документа:** {doc_annotation}'
        if doc_annotation[-1] not in {'.', ',', '?', '!', ':', ';'}:
            new_document_description += '.'
    else:
        new_document_description += '\n\nДокумент без аннотации.'
    del doc_title, doc_annotation
    local_indices_of_relevant_chunks = sorted(chunk_indices)
    num_relevant_chunks = len(local_indices_of_relevant_chunks)
    if num_relevant_chunks > 1:
        num_relevant_chunks_as_str = str(num_relevant_chunks)
        new_document_description += f'\n\n**{num_relevant_chunks_as_str} наиболее '
        if num_relevant_chunks_as_str[-1] == '1':
            new_document_description += 'релеватный фрагмент документа:**'
        elif num_relevant_chunks_as_str[-1] in {'2', '3', '4'}:
            new_document_description += 'релеватных фрагмента документа:**'
        else:
            new_document_description += 'релеватных фрагментов документа:**'
        del num_relevant_chunks_as_str
        for idx, val in enumerate(local_indices_of_relevant_chunks):
            chunk_boundaries = doc_chunks[val]
            chunk_text = doc_text[chunk_boundaries['start_char']:chunk_boundaries['end_char']]
            new_document_description += f'\n\n<Фрагмент {idx + 1}>\n{chunk_text}\n</Фрагмент {idx + 1}>'
    else:
        chunk_boundaries = doc_chunks[local_indices_of_relevant_chunks[0]]
        chunk_text = doc_text[chunk_boundaries['start_char']:chunk_boundaries['end_char']]
        new_document_description += f'\n\n**Один самый релеватный фрагмент документа:**\n\n{chunk_text}'
    del doc_text, doc_chunks
    return new_document_description


def prepare_relevant_documents(indices_of_relevant_chunks: List[int], scores_of_relevant_chunks: List[float],
                               documents: List[Dict[str, Union[str, float, List[Dict[str, int]]]]],
                               chunk_mapping: Dict[str, Dict[str, int]],
                               min_document_quality: float) -> Dict[int, Dict[str, Union[float, List[int]]]]:
    selected_documents = dict()
    for global_index, relevance in zip(indices_of_relevant_chunks, scores_of_relevant_chunks):
        document_index, local_chunk_index = global_chunk_index_to_local(global_index, documents, chunk_mapping)
        if documents[document_index]['quality_score'] >= min_document_quality:
            if document_index in selected_documents:
                if relevance > selected_documents[document_index]['relevance']:
                    selected_documents[document_index]['relevance'] = relevance
                selected_documents[document_index]['chunks'].append(local_chunk_index)
            else:
                selected_documents[document_index] = {'relevance': relevance, 'chunks': [local_chunk_index]}
    return selected_documents


def format_document_reference(document_index: int, number: str, max_number_width: int,
                              documents: List[Dict[str, Union[str, float, List[Dict[str, int]]]]]) -> str:
    doc_urls = normalize_urls(documents[document_index].get('url', []))
    doc_title = ' '.join(documents[document_index]['doc_title'].strip().split()).strip()
    if len(doc_urls) > 0:
        url_block = ('\n' + ''.join([' ' for _ in range(max_number_width)])).join(doc_urls)
        if len(doc_title) > 0:
            new_reference = number + doc_title + '\n'
            new_reference += ''.join([' ' for _ in range(max_number_width)])
            new_reference += url_block
        else:
            new_reference = number + url_block
    else:
        new_reference = number + doc_title
    return new_reference


def prepare_context(indices_of_relevant_chunks: List[int], scores_of_relevant_chunks: List[float],
                    documents: List[Dict[str, Union[str, float, List[Dict[str, int]]]]],
                    chunk_mapping: Dict[str, Dict[str, int]],
                    min_document_quality: float) -> Tuple[List[str], List[str]]:
    selected_documents = prepare_relevant_documents(
        indices_of_relevant_chunks, scores_of_relevant_chunks,
        documents, chunk_mapping,
        min_document_quality
    )
    if len(selected_documents) == 0:
        return [], []
    descriptions_of_selected_documents = []
    descriptions_of_urls = []
    ordered_indices_of_selected_documents = sorted(
        list(selected_documents.keys()),
        key=lambda idx: selected_documents[idx]['relevance']
    )
    max_number_width = len(f'{len(ordered_indices_of_selected_documents)}. ')
    for counter, document_index in enumerate(ordered_indices_of_selected_documents):
        descriptions_of_selected_documents.append(document_to_text(
            document_index=document_index,
            chunk_indices=selected_documents[document_index]['chunks'],
            documents=documents
        ))
        number = f'{counter + 1}. '
        while len(number) < max_number_width:
            number += ' '
        descriptions_of_urls.append(format_document_reference(
            document_index=document_index,
            number=number,
            max_number_width=max_number_width,
            documents=documents
        ))
        del number
    return descriptions_of_selected_documents, descriptions_of_urls


def graph_element_to_text(item: Dict[str, Any],
                          documents: List[Dict[str, Union[str, float, List[Dict[str, int]]]]],
                          chunk_mapping: Dict[str, Dict[str, int]]) -> str:
    source_titles = []
    for cur_global_index in item.get('sources', []):
        try:
            document_index, _ = global_chunk_index_to_local(cur_global_index, documents, chunk_mapping)
        except ValueError:
            continue
        cur_title = ' '.join(documents[document_index]['doc_title'].strip().split()).strip()
        if len(cur_title) == 0:
            cur_title = 'Документ без названия.'
        if cur_title not in source_titles:
            source_titles.append(cur_title)
    if item.get('kind') == 'relation':
        new_element_description = '**Элемент графа знаний: связь.**\n\n'
        new_element_description += (f"**Связь:** {item.get('subject')} "
                                    f"—[{item.get('relation_type')}]→ {item.get('object')}.")
    else:
        new_element_description = '**Элемент графа знаний: сущность.**\n\n'
        new_element_description += f"**Название:** {item.get('name')} (тип: {item.get('type')})."
    element_description = ' '.join(str(item.get('description', '')).strip().split()).strip()
    if len(element_description) > 0:
        new_element_description += f'\n\n**Описание:** {element_description}'
        if element_description[-1] not in {'.', ',', '?', '!', ':', ';'}:
            new_element_description += '.'
    if len(source_titles) > 0:
        new_element_description += ('\n\n**Встречается во фрагментах документов:** '
                                    + '; '.join(source_titles) + '.')
    return new_element_description


def prepare_graph_elements_context(items: List[Dict[str, Any]],
                                   documents: List[Dict[str, Union[str, float, List[Dict[str, int]]]]],
                                   chunk_mapping: Dict[str, Dict[str, int]]) -> Tuple[List[str], List[int]]:
    """
    Renders the graph elements (entities and relations) of the united search pool into
    self-describing text blocks for the QA context, and collects the indices of the
    knowledge base documents that the element source chunks belong to.
    """
    element_texts = []
    referenced_document_indices = set()
    for cur_item in items:
        element_texts.append(graph_element_to_text(cur_item, documents, chunk_mapping))
        for cur_global_index in cur_item.get('sources', []):
            try:
                document_index, _ = global_chunk_index_to_local(cur_global_index, documents, chunk_mapping)
            except ValueError:
                continue
            referenced_document_indices.add(document_index)
    return element_texts, referenced_document_indices
