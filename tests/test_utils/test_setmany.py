from dataclasses import dataclass, field
from typing import Dict, List, Union

from smol.utils.setmany import SetMany


@dataclass
class A:
    index: int
    num_threads: int


@dataclass
class Container:
    many: Union[List[A], Dict[int, A]]
    num_threads: int = field(default=SetMany("num_threads", "many"))


def test_set_many():
    # test with list
    many = [A(i, 1) for i in range(10)]
    test = Container(many, 1)

    assert test.num_threads == 1
    for a in test.many:
        assert a.num_threads == 1

    test.num_threads = 2

    assert test.num_threads == 2
    for a in test.many:
        assert a.num_threads == 2

    # test with dict
    many = {i: A(i, 1) for i in range(10)}
    test = Container(many, 1)

    assert test.num_threads == 1
    for a in test.many.values():
        assert a.num_threads == 1

    test.num_threads = 2

    assert test.num_threads == 2
    for a in test.many.values():
        assert a.num_threads == 2
