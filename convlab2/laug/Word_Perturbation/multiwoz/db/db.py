from typing import Union, Callable, List, Optional
from ..util import choice


class BaseDB:
    ...


class DB(list, BaseDB):
    """
    DB is a list of dicts.
    """

    def query(self, conditions: Union[dict, Callable[[dict], bool], None]) -> List[dict]:
        if conditions is None:
            return self
        assert callable(conditions) or isinstance(conditions, dict)
        if isinstance(conditions, dict):
            fn = lambda item: all(item[k] == conditions[k] for k in conditions if k in item)
        else:
            fn = conditions
        return [item for item in self if fn(item)]

    def sample(self, conditions=None) -> Optional[dict]:
        list_ = self.query(conditions)
        if list_:
            try:
                return choice(list_)
            except (IndexError, ValueError):
                pass
        return None
