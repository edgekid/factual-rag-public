from collections import defaultdict

class Ucea:
    def __init__(self, idx, match, parent=None):
        self.idx = idx
        self.match = match
        self.parent = parent
        self.delete_flag = False
        # todo
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.match == other.match
        elif isinstance(other, int):
            return self.match == other
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        if self.parent is None:
            return 1
        return 1 + len(self.parent)

    def __repr__(self):
        return str(self)

    def __str__(self):
        if self.parent is not None:
            return str(self.parent) + ' (' + str(self.idx) + ', ' + str(self.match) + ')'
        return '(' + str(self.idx) + ', ' + str(self.match) + ')'

    def first(self):
#        if self.parent is None:
#            return self
        fs = self
        while fs.parent is not None:
            fs = fs.parent
#        return self.parent.first()
        return fs

def find_gunoi(val, container):
    return next((x for x in container if val == x), None)


def best_masa11(match, prev, pprev):
    def verif1(res):
        if res.parent is None:
            return False
        return res.parent.match == res.match-1
        # if res.match-1 in prev:
        #     return True
        # a=find_gunoi(res.match-1,pprev)
        # if a is None:
        #     return False
    def verif2(res):
        if res.parent is None:
            return False
        return res.parent.idx == res.idx-1

    # a[i-1]=b[j-2]
    res1 = find_gunoi(match-2, prev)
    # a[i-2]=b[j-1]
    res2 = find_gunoi(match-1, pprev)

    # a[i-1]=b[j-2]
    if res1 is not None and not verif1(res1):
        res1 = None

    # a[i-1]=b[j-2]
    if res2 is not None and not verif2(res2):
        res2 = None

    # choose best
    if res1 is not None and res2 is not None:
        return res1 if len(res1) >= len(res2) else res2

    if res1 is not None:
        return res1

    if res2 is not None:
        return res2

    return None

def create_fuzzy_match(idx, match, prev, pprev):
    # a[i-1]=b[j-1]
    parent= find_gunoi(match - 1, prev)
    if parent is not None:
        ret = Ucea(idx, match, parent)
        return ret

    # a[i-1]=b[j-2] or a[i-2]=b[j-1]
    if match - 2 in prev or match - 1 in pprev:
        parent = best_masa11(match, prev, pprev)
        if parent is not None:
            return Ucea(idx, match, parent)

    # a[i-2]=b[j-2]
    parent= find_gunoi(match - 2, pprev)
    if parent is not None and parent.parent is not None and parent.parent.match == match - 3 and parent.parent.idx == idx - 3:
        ret = Ucea(idx, match, parent)
        return ret

    # Trivial parent
    t = Ucea(idx - 1, match - 1, None)
    return Ucea(idx, match, t)

def charlen(ucea, short):
#    if ucea is None:
#        return 0

    length = 0
    while ucea is not None:
        length += len(short[ucea.idx])
        ucea = ucea.parent
#    return len(short[ucea.idx])+charlen(ucea.parent,short)
    return length

def print_match(ucea, short):
    str = ''
    while ucea is not None and  ucea.parent is not None:
        str = short[ucea.idx] + ' ' + str
        ucea = ucea.parent
    return str

def MATCH(short: list[str], long_: list[str], tau: int):
    # Create hashtable for fast word to index lookup
    A = defaultdict(list)
    for idx, word in enumerate(short):
        A[word].append(idx)

    # Find all matches in long for every word in short
    B = [[] for _ in range(len(short))]
    for long_idx, word in enumerate(long_):
        for short_idx in A[word]:
            B[short_idx].append(long_idx)

    # Iteratively build fuzzy matches based on single word matches
    Sequences = [[] for _ in range(len(short))]
    for idx, matches in enumerate(B):
        prev = Sequences[idx - 1] if idx > 0 else []
        pprev = Sequences[idx - 2] if idx > 1 else []
        for match in matches:
            #print("{}, {}:".format(idx, match))
            seq = create_fuzzy_match(idx, match, prev, pprev)
            #print(seq)
            if seq is not None:
                Sequences[idx].append(seq)
            # Remove seq.parent later as there is now a longer match
            # containing seq.parent. We cannot delete now as seq.parent
            # might be needed for other fuzzy matches
            if seq.parent is not None:
                seq.parent.delete_flag = True

    # Delete short matches or subsequences of longer matches
    sum = 0
    best_match = None
    for seqs in Sequences:
        for seq in seqs:
            #print(seq)
            #chlen = charlen(seq, short) - 1 # Exclude trivial parent
            #if chlen < tau or seq.delete_flag:
            #    del seq
            #else:
            #    print(chlen)
            #    sum += chlen**2
            length = len(seq) - 1 # Exclude trivial parent
            if length < tau or seq.delete_flag:
                del seq
            else:
                sum += length ** 2
                if best_match == None or len(best_match) - 1 < length:
                    best_match = seq
                
#    total_len = (len(short) ** 2) * (len(long_) - len(short) + 1)
#    return sum / total_len
    return sum, print_match(best_match, short)

if __name__=='__main__':
    #short=['a','b','c','d']
    #long=['a','b','c','d','e','b','c','a','d','a','b','d','e','a','e','e','b']
    short=['a','b','c','d','a']
    long=['a','b','c','a','e','b','c','a','d','a']
    print(MATCH(short, long, 2))

