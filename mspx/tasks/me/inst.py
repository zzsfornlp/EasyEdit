#

# instances

from copy import deepcopy

__all__ = [
    "Item", "Instance",
]

# -----
# new format to make it more flexible
"""
# -- full format
edit: list[dict]  # list of edits for one instance
paraphrase: list[dict]  # simple cases of paraphrasing
portability: list[dict]  # portability testing
locality: list[dict]  # locality testing
-> one item (some may be optional for some fields):
    subject: str  # subject item
    type: str  # type of this one item
    question: str  # question or prompt
    answer: str  # new target answer
    answer_alias: list[str]  # also potential answers
    answer_old: str  # old answer
    answer_old_alias: list[str]  # old alternative answers
# -- each editing item needs:
    subject, question, answer -> subject, prompt, target_new
"""
# -----

class BaseInst:
    def __init__(self, **kwargs):
        self.from_json(kwargs)

    def __getitem__(self, item):
        return self.__dict__[item]

    def to_json(self):
        ret = {}
        for k, v in self.__dict__.items():
            ret[k] = v.to_json() if isinstance(v, BaseInst) else deepcopy(v)
        return ret

    def from_json(self, d):
        for k, v in d.items():
            v0 = getattr(self, k)
            if isinstance(v0, BaseInst):
                v0.from_json(v)
            else:
                setattr(self, k, v)

    @classmethod
    def create(cls, _d=None, **kwargs):
        if _d is None:
            _d = {}
        if kwargs:
            _d.update(kwargs)
        return cls(**_d)

class Item(BaseInst):
    def __init__(self, **kwargs):
        self.type: str = ""  # type of this one item
        self.subject: str = ""  # subject item
        self.question: str = ""  # question or prompt
        self.answer: str = ""  # new target answer
        self.answer_alias: list[str] = []  # also potential answers
        self.answer_old: str = ""  # old answer
        self.answer_old_alias: list[str] = []  # old alternative answers
        super().__init__(**kwargs)

    def format(self):
        d = {"subject": self.subject, "prompt": self.question, "target_new": self.answer}  # use self!
        return self.format_dict(d)

    @classmethod
    def format_dict(cls, d):
        pieces = [d['prompt'], d['target_new']]
        pieces = [z.strip() for z in pieces if z.strip()]
        ret = " ".join(pieces)
        return ret


class Instance(BaseInst):
    def __init__(self, **kwargs):
        self.edit: list[Item] = []
        self.rephrase: list[Item] = []
        self.portability: list[Item] = []
        self.locality: list[Item] = []
        self.info = {}  # extra information
        super().__init__(**kwargs)

    def to_json(self):
        ret = {k: ([z.to_json() for z in v] if isinstance(v, list) else v) for k, v in self.__dict__.items()}
        return ret

    def from_json(self, d):
        for k, v in d.items():
            setattr(self, k, ([Item.create(**z) for z in v] if isinstance(v, list) else v))

    def get_edit_insts(self):  # for previous format!
        ret = [{"subject": _item.subject, "prompt": _item.question, "target_new": _item.answer} for _item in self.edit]
        return ret
