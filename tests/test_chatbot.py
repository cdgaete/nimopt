from chatbot import BEGIN, END, RULES, needles, pages, system_prompt


def test_the_prompt_is_the_fixed_rules_around_the_whole_corpus():
    prompt = system_prompt()
    assert prompt.startswith(RULES)
    assert prompt.count(BEGIN) == 1
    assert prompt.count(END) == 1
    inside = prompt[prompt.index(BEGIN) : prompt.index(END)]
    missing = [route for route, _ in pages() if f"# {route}\n" not in inside]
    assert missing == [], missing


def test_the_prompt_is_byte_identical_across_builds():
    assert system_prompt() == system_prompt()


def test_the_corpus_splits_into_distinct_routed_pages():
    book = pages()
    assert len(book) >= 40
    assert len({route for route, _ in book}) == len(book)
    assert all(route.startswith("/") for route, _ in book)


def test_needles_name_exactly_one_page_each():
    book = pages()
    pairs = needles()
    assert len(pairs) >= 12
    for route, token in pairs:
        assert route != "/"
        assert sum(token in body for _, body in book) == 1


def test_needles_spread_across_the_corpus():
    book = pages()
    at = {route: index for index, (route, _) in enumerate(book)}
    places = sorted(at[route] for route, _ in needles())
    assert places[0] <= len(book) // 3
    assert places[-1] >= 2 * len(book) // 3
