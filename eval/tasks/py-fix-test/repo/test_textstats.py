from textstats import word_count, most_common_word, average_word_length


def test_word_count():
    assert word_count("a b  c") == 3
    assert word_count("") == 0


def test_most_common_word_ties_first_seen():
    # 'the' and 'cat' both appear twice; 'the' was seen first
    assert most_common_word("the cat and the cat!") == "the"


def test_average_word_length():
    assert average_word_length("ab cd") == 2.0
    assert average_word_length("") == 0.0
