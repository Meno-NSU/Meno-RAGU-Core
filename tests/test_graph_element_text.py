from context_utils.context_utils import graph_element_to_text


def relation(relation_type):
    return {"kind": "relation", "subject": "Иван Петров", "object": "НГУ", "relation_type": relation_type,
            "description": "Иван Петров выступил с докладом в НГУ.", "sources": []}


def test_a_relation_shows_its_type():
    text = graph_element_to_text(relation("WORKPLACE"), [], {})
    assert "**Связь:** Иван Петров —[WORKPLACE]→ НГУ." in text


def test_a_relation_whose_type_does_not_fit_shows_the_link_and_its_description_without_the_type():
    """Graph building keeps a relation whose type does not fit its ends as UNFIT:<type> (no text is lost); the answer
    prompt has no use for the mark, and the unfit type would state a wrong fact."""
    text = graph_element_to_text(relation("UNFIT:PARTICIPANT_IN"), [], {})
    assert "UNFIT" not in text and "PARTICIPANT_IN" not in text
    assert "**Связь:** Иван Петров — НГУ." in text
    assert "**Описание:** Иван Петров выступил с докладом в НГУ." in text


def test_a_relation_retyped_to_related_to_shows_the_link_without_a_type():
    """Graph v5 retypes a relation whose fact no type of the ontology names to RELATED_TO; its text states the fact,
    and the generic type says nothing the answer prompt could use."""
    text = graph_element_to_text(relation("RELATED_TO"), [], {})
    assert "RELATED_TO" not in text
    assert "**Связь:** Иван Петров — НГУ." in text
