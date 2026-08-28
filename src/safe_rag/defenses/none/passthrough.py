from safe_rag.defenses.base import PassthroughDefense


class NoneDefense(PassthroughDefense):
    name = "none"
