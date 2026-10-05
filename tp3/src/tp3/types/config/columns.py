from typing import List, TypedDict

class Columns(TypedDict):
    input: List[str] | str
    output: List[str] | str
    expected: List[str] | str
