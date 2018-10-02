
class Entity:
    def __init__(self):
        self.id = None
        self.type = None
        self.start = None
        self.end = None
        self.text = None
        self.sent_idx = None
        self.tf_start = None
        self.tf_end = None
        self.norm_id = None

    def create(self, id, type, start, end, text, sent_idx, tf_start, tf_end):
        self.id = id
        self.type = type
        self.start = start
        self.end = end
        self.text = text
        self.sent_idx = sent_idx
        self.tf_start = tf_start
        self.tf_end = tf_end

    def append(self, start, end, text, tf_end):

        whitespacetoAdd = start - self.end
        for _ in range(whitespacetoAdd):
            self.text += " "
        self.text += text

        self.end = end
        self.tf_end = tf_end

    def getlength(self):
        return self.end-self.start

    def equals(self, other):
        if self.type == other.type and self.start == other.start and self.end == other.end:
            return True
        else:
            return False

    def equals_span(self, other):
        if self.start == other.start and self.end == other.end:
            return True
        else:
            return False


class Document:
    def __init__(self):
        self.entities = None
        self.sentences = None
        self.name = None



