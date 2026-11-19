import graph_v61_collections as build
from graph_utils import graph_patch as gp


def test_point_ids_follow_the_rule_of_ragu():
    collection = '_home_m.kulakov_meno-v2_ragu-v2_bundle_knowledge_ragu_graph_v6_vdb_entity'
    assert build.point_id(collection, 'ent-af2182031f5e3f3062ae19e7f840ffaa') == '000013c8-41dd-5486-bd73-dcb08fb20321'
    assert build.point_id(collection + '_v61', 'ent-af2182031f5e3f3062ae19e7f840ffaa') != '000013c8-41dd-5486-bd73-dcb08fb20321'


def test_unchanged_texts_take_their_vectors_and_changed_ones_are_embedded():
    old = {'a': 'А - один', 'b': 'Б - два', 'c': 'В - три'}
    new = {'a': 'А - один', 'b2': 'Б - два', 'c': 'В - три и четыре', 'd': 'Г - пять'}
    assert build.classify(new, old) == ({'a': 'a', 'b2': 'b'}, ['c', 'd'])


def test_payloads_have_the_fields_of_the_v6_collections():
    a, b = gp.Node('ent-1', 'А', 'PERSON', 'Человек.', ['c1', 'c2']), gp.Node('ent-2', 'НГУ', 'UNIVERSITY', 'Вуз.', ['c1'])
    text, payload = build.entity_item(a, chunks_as_text=True)
    assert text == 'А - Человек.' and payload['source_chunk_id'] == "['c1', 'c2']" and payload['id'] == payload['__ragu_id__'] == 'ent-1'
    assert set(payload) == {'entity_name', 'entity_type', 'description', 'source_chunk_id', 'documents_id', 'clusters', 'id', '__ragu_id__'}
    text, payload = build.relation_item(gp.Edge('rel-1', 'ent-1', 'ent-2', 'WORKPLACE', 'Работает в НГУ.', 5.0, ['c1']), {'ent-1': a, 'ent-2': b})
    assert text == 'Работает в НГУ.' and payload['subject_name'] == 'А' and payload['object_name'] == 'НГУ' and payload['source_chunk_id'] == ['c1']
