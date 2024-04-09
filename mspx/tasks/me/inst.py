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

    def __repr__(self):
        return repr(self.__dict__)

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

PROMPT_TEMPLATES = {
    "orig": lambda x: x,
    "qa": lambda x: f"Question: {x}\nAnswer:",
}

class Item(BaseInst):
    PROMPT_TEMPLATE_F = (lambda x: x)

    def __init__(self, **kwargs):
        self.type: str = ""  # type of this one item
        self.subject: str = ""  # subject item
        self.question: str = ""  # question or prompt
        self.answer: str = ""  # new target answer
        self.answer_alias: list[str] = []  # also potential answers
        self.answer_old: str = ""  # old answer
        self.answer_old_alias: list[str] = []  # old alternative answers
        super().__init__(**kwargs)

    def get_alt_dict(self):
        d = {"subject": self.subject, "prompt": self.prompt, "target_new": self.answer}  # use self!
        return d

    def format(self):
        d = self.get_alt_dict()
        return self.format_dict(d)

    @property
    def prompt(self):
        return Item.PROMPT_TEMPLATE_F(self.question)

    @classmethod
    def format_dict(cls, d):
        pieces = [d['prompt'], d['target_new']]
        pieces = [z.strip() for z in pieces if z.strip()]
        ret = " ".join(pieces)
        return ret

    @classmethod
    def set_prompt_template(cls, template: str):
        cls.PROMPT_TEMPLATE_F = PROMPT_TEMPLATES[template]

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
        ret = [_item.get_alt_dict() for _item in self.edit]
        return ret
