import random

from prop import AtomForm, NegForm, ConForm, DisForm, ImpForm, Form


# region generation procedure from DEEP LEARNING FOR SYMBOLIC MATHEMATICS

def memoize(f):
    cache = {}

    def decorate(*args):
        if args in cache:
            return cache[args]
        else:
            cache[args] = f(*args)
            return cache[args]

    return decorate


@memoize
def D(e, n):
    if e == 0:
        return 0
    elif n == 0:
        return 1
    else:
        return D(e - 1, n) + D(e, n - 1) + D(e + 1, n - 1)


def PL(e, n, a, k):
    if a == 1:
        return D(e - k, n - 1) / D(e, n)
    elif a == 2:
        return D(e - k + 1, n - 1) / D(e, n)
    else:
        return 0.


def PK(e, n, k):
    return D(e - k + 1, n - 1) / D(e, n)


import random


def sampleL(e, n):
    while True:
        s = random.uniform(0., 1.)
        for a in [1, 2]:
            for k in range(0, e, 1):
                prob = PL(e, n, 1, k)
                if s < prob:
                    return a, k
                s -= prob


def sampleK(e, n):
    while True:
        s = random.uniform(0., 1.)
        for k in range(0, e, 1):
            prob = PK(e, n, k)
            if s < prob:
                return k
            s -= prob


class node:
    def __init__(self):
        '''
        none=empty node
        []=leaf
        '''
        self.childs = None
        self.lbl = 'empty'

    def set_arity(self, ar):
        for _ in range(ar):
            self.childs.append(node())

    def insert_first(self, ar, lbl):
        if self.childs is None:
            self.childs = [node() for _ in range(ar)]
            self.lbl = lbl
            return self
        else:
            for c in self.childs:
                r = c.insert_first(ar, lbl)
                if r is not None:
                    return r
            return None

    def print(self, t=0, sp=1):
        print(' ' * t * sp + self.lbl)
        if self.childs is not None:
            for c in self.childs:
                c.print(t + 1, sp)

    def has_empty_nodes(self):
        if self.childs is None:
            return True
        for c in self.childs:
            if c.has_empty_nodes():
                return True
        return False


def makeubtree(n):
    e = 1
    t = node()
    while n > 0:
        a, k = sampleL(e, n)
        for _ in range(k):
            t.insert_first(0, None)  # 'leaf')
        if a == 1:
            op = 'neg'
            t.insert_first(1, op)
            e = e - k
        else:
            op = None  # random.choice(['con', 'dis'])
            t.insert_first(2, op)
            e = e - k + 1
        n = n - 1
    while t.has_empty_nodes():
        t.insert_first(0, 'leaf')
    return t


# endregion


# назначение переменных


class expression_tree:
    def __init__(self):
        self.root = [None, None, None]
        self.branches_i = 0
        self.branch_var_dict = {}
        self.vars = set()

    def init_operations(self):
        def iopsrec(r):
            if r[0] is None and r[1] is None:
                r[2] = 'b{:0>4d}'.format(self.branches_i)
                self.branch_var_dict[r[2]] = None
                self.branches_i += 1
            elif r[0] is None and r[1] is not None:
                r[2] = 'N'
            elif r[0] is not None and r[1] is None:
                r[2] = 'N'
            else:
                r[2] = random.choice(['C', 'D', 'I'])
            if r[0] is not None:
                iopsrec(r[0])
            if r[1] is not None:
                iopsrec(r[1])

        iopsrec(self.root)

    def contains_word(self, s):
        r = self.root
        for i, b in enumerate(s):
            if r[b] is None:
                return False
            r = r[b]
        return True

    def insert_word(self, s):
        r = self.root
        for i, b in enumerate(s):
            # if r[2]=='b':
            #     return False
            if b in [0, 1]:
                if r[b] is None:
                    r[b] = [None, None, None]
                r = r[b]
            else:
                r[2] = 'b'
                return True

    def tolist(self):
        def tlr(prefx, r):
            rs = []
            if r[0] is None and r[1] is None:
                return [prefx]
            if r[0] is not None:
                rs.extend(tlr(prefx + [0], r[0]))
            if r[1] is not None:
                rs.extend(tlr(prefx + [1], r[1]))
            return rs

        return tlr([], self.root)

    def assign_variables(self, n):
        while n > len(self.branch_var_dict):
            n = n // 2
        # if n <= len(self.branch_var_dict):

        vs = [x for x in range(1, n + 1, 1)]
        # f = True
        # while f:
        #     crp = random.choices(vs, k=len(self.branch_var_dict))
        #     f = len(set(crp)) < len(vs)
        # for k, v in zip(self.branch_var_dict.keys(), crp):
        #     self.branch_var_dict[k] = 'v' + str(v)
        #     self.vars.add('v' + str(v))

        branches = list(self.branch_var_dict.keys())
        distrib = {v: [] for v in vs}
        for v in distrib:
            if len(branches) == 0:
                break
            bi = random.randint(0, len(branches) - 1)
            b = branches.pop(bi)
            distrib[v] = [b]
        for b in branches:
            distrib[random.choice(vs)].append(b)
        for v in distrib:
            distrib[v].sort(key=str)
        distrib = sorted(distrib.values(), key=lambda x: x[0])
        for v, l in enumerate(distrib):
            vname = 'v{}'.format(v + 1)
            self.vars.add('v' + str(v))
            for k in l:
                self.branch_var_dict[k] = vname

    def remove_double_negations(self):
        def rdn(r):
            # if r is None:
            #     return None
            if r[2] in ['C', 'I', 'D']:
                return [rdn(r[0]), rdn(r[1]), r[2]]
            else:
                if r[2] != 'N':
                    return r
                else:
                    c = r[0] if r[0] is not None else r[1]

                    if c[2] != 'N':
                        if c[2] not in ['C', 'D', 'I']:
                            return r
                        # print(r[2], c[2])
                        return [[rdn(c[0]), rdn(c[1]), c[2]], None, 'N']
                    else:
                        c1 = c[0] if c[0] is not None else c[1]
                        return rdn(c1)

        self.root = rdn(self.root)

    def ToFormula(self):
        def tfr(r):
            if r[2] == 'I':
                return ImpForm(tfr(r[0]), tfr(r[1]))
            elif r[2] == 'C':
                return ConForm(tfr(r[0]), tfr(r[1]))
            elif r[2] == 'D':
                return DisForm(tfr(r[0]), tfr(r[1]))
            elif r[2] == 'N':
                c = r[0] if r[0] is not None else r[1]
                return NegForm(tfr(c))
            else:
                return AtomForm(self.branch_var_dict[r[2]])

        return tfr(self.root)

    def __len__(self):
        def count_subtree(r, is_root=False):
            if r is None:
                return 0
            elif r[0] is None and r[1] is None:
                return 0 if is_root else 1
            else:
                return count_subtree(r[0]) + count_subtree(r[1])

        return count_subtree(self.root, True)

    def __iter__(self):
        return (x for x in self.tolist())

    def __repr__(self):
        def recs(r):
            if r[2] == 'N':
                if r[0] is not None:
                    return 'N' + recs(r[0])
                else:
                    return 'N' + recs(r[1])
            elif r[2] in ['I', 'C', 'D']:
                return '{}{}{}'.format(r[2], recs(r[0]), recs(r[1]))
            else:
                return r[2] if self.branch_var_dict[r[2]] is None else self.branch_var_dict[r[2]]

        return recs(self.root)


def node2lists(n: node):
    r = [None, None, None]
    r[2] = n.lbl
    if n.childs is not None:
        for i, c in enumerate(n.childs):
            if c is not None:
                r[i] = node2lists(c)
    return r


def node2expression_tree(n: node):
    t = expression_tree()
    t.root = node2lists(n)
    return t


def generate_branches(n, s):
    res = expression_tree()
    while len(res) < n:

        nb = []
        f = True
        while f:
            b = random.choices([0, 1, 'b'], [0.5 - s / 2, 0.5 - s / 2, s], k=1)[0]
            nb.append(b)
            if b == 'b':
                f = False

        if len(nb) < 2:
            continue
        # print(nb)
        res.insert_word(nb)
    return res


def consistent_join(liters, newlitera):
    # nr = newlitera.reduce()
    nr = newlitera
    if isinstance(nr, AtomForm):
        nn = NegForm(newlitera)
        # nr = newlitera.reduce()
        for l in liters:
            # print('cmp',l,nn)
            if str(l) == str(nn):
                return None
            if str(l) == str(nr):
                return liters
        return liters + [nr]
    elif isinstance(nr, NegForm):
        nn = nr.value
        # nr = newlitera.reduce()
        for l in liters:
            # print('cmp',l,nn)
            if str(l) == str(nn):
                return None
            if str(l) == str(nr):
                return liters
        return liters + [nr]
    elif type(nr) is ConForm:
        res = liters.copy()
        for a in newlitera.members:
            res = consistent_join(res, a)
            if res is None:
                break
        return res
    else:
        raise ValueError()


def a_tableux(formulas, liters=None):
    if liters is None:
        liters = []

    def select_formula(formulas: list):
        if len(formulas) == 0:
            return None
        for i, f in enumerate(formulas):
            if type(f) is AtomForm or type(f) is NegForm and type(f.value) is AtomForm:
                return i
        for i, f in enumerate(formulas):
            if type(f) is ConForm:
                return i
        for i, f in enumerate(formulas):
            if type(f) is DisForm:
                return i
        return 0

    if len(formulas) == 0:
        return [liters]
    res = []
    locfs = formulas.copy()
    fi = select_formula(locfs)
    f = locfs[fi]
    locfs.pop(fi)
    # print('f:', f, 'forms:', locfs, 'lits:', liters)
    if isinstance(f, ConForm):
        res = a_tableux([f.a, f.b] + locfs, liters)
    elif isinstance(f, AtomForm):
        j = consistent_join(liters, f)
        if not j is None:
            res.extend(a_tableux(locfs, j))
        else:
            return []
    elif isinstance(f, DisForm):
        for m in [f.a, f.b]:
            res.extend(a_tableux([m] + locfs, liters))
    elif isinstance(f, ImpForm):
        for m in [NegForm(f.a), f.b]:
            res.extend(a_tableux([m] + locfs, liters))
    elif isinstance(f, NegForm):
        # sf = f.reduce()
        v = f.value
        if isinstance(v, ConForm):
            for m in [NegForm(v.a), NegForm(v.b)]:
                res.extend(a_tableux([m] + locfs, liters))
        elif isinstance(v, AtomForm):
            j = consistent_join(liters, f)
            if not j is None:
                res.extend(a_tableux(locfs, j))
            else:
                return []
        elif isinstance(v, DisForm):
            res = a_tableux([NegForm(v.a), NegForm(v.b)] + locfs, liters)
        elif isinstance(v, ImpForm):
            res = a_tableux([v.a, NegForm(v.b)] + locfs, liters)
        elif isinstance(v, NegForm):
            res = a_tableux([v.value] + locfs, liters)
        else:
            return []

    else:
        return []
    # print('res:', res)
    return res


def simplify(kb: list):
    def is_strong_subset(a, b):  # a - подмножество b, a не равно b
        if len(a) >= len(b):
            return False
        f = True
        for ae in a:
            f = ae in b  # long working because of repr
            if not f: break
        return f  # and len(a) < len(b)

    def minimal(f, kb):
        fl = True
        for f2 in kb:
            fl = not is_strong_subset(f2, f)
            if not fl: break
        return fl

    kb1 = [tuple(sorted(x, key=str)) for x in kb]
    res = set()
    # kb = kb.copy()
    # while len(kb) > 0:
    #     cand = kb.pop(0)
    #     if minimal(cand, kb):
    #         res.append(cand)
    for cand in kb1:
        if minimal(cand, kb1):
            res.add(cand)
    return [list(x) for x in res]


# print(a_tableux([Form.parse_formula('~(a=>b)')]))

def neg_con(br):
    if len(br) == 1:
        sbf = NegForm(br[0])
    elif len(br) == 2:
        sbf = NegForm(DisForm(br[0], br[1]))
    else:
        sbf = NegForm(DisForm(br[0], neg_con(br[1:]).value))
    return sbf


# N = 20
# s = 0.1
# bn = 10
# vs = 4
#
# alg = 2


def simple_generation(bn, alg, vs, s, N):
    for _ in range(N):
        print('=================================================================================')
        if alg == 1:
            ag = generate_branches(bn, s)
        else:
            t = makeubtree(int(bn * 1.5))
            ag = node2expression_tree(t)
        print(ag.tolist())
        ag.init_operations()
        ag.assign_variables(vs)
        print('generated formula')
        print(ag)
        f = ag.ToFormula()
        print(f.prefixstr())
        print(f)
        print('open branches')
        print(a_tableux([f]))
        ag.remove_double_negations()
        print('removed double negations')
        print(ag)
        f = ag.ToFormula()
        print(f.prefixstr())
        print(f)
        open_branches = a_tableux([f])
        print(open_branches)
        opbr = simplify(open_branches)
        print('minimized branches')
        print(opbr)
        if len(open_branches) == 0:
            print('UNSAT')
            continue
        print('SAT')

        contras = list({random.choice(x) for x in open_branches})
        print('contras')
        print(contras)
        neg_sel_br = neg_con(contras)
        closed_f = ConForm(f, neg_sel_br)
        print('closed formula')
        print(closed_f)
        print('branches:')
        open_branches = a_tableux([closed_f])
        print(open_branches)


def reduce(a: Form):
    if isinstance(a, NegForm) and isinstance(a.value, NegForm):
        return a.value.value
    else:
        return a


def close_open_branches(brs):
    sbrs = sorted(brs, key=len)
    res_lst = []
    for literals in sbrs:
        covered = [x for x in literals if x in res_lst]
        if len(covered) > 0:
            continue
        c1 = [x for x in literals if reduce(NegForm(x)) not in res_lst]
        c2 = [x for x in literals if reduce(NegForm(x)) in res_lst]
        if len(c1) > 0:
            na = random.choice(c1)
            res_lst.append(na)
        elif len(c2) > 0:
            na = random.choice(c2)
            res_lst.append(na)
            break

    return res_lst  # list({random.choice(x) for x in brs})


def generate_dataset(alg, N, bn, vs, s, ops, remove_dn):
    sats = set()
    unstas = set()

    while len(sats) < N // 2:
        print('=================================================================================')
        if alg == 1:
            ag = generate_branches(bn, s)
        else:
            t = makeubtree(ops)
            ag = node2expression_tree(t)
        # print(ag.tolist())
        ag.init_operations()
        ag.assign_variables(vs)
        print('generated formula:', ag)
        # print(f.prefixstr())
        # print('infix form:', f)
        if remove_dn:
            ag.remove_double_negations()
            print('removed double negations')
            print(ag)
        f = ag.ToFormula()
        print('prefix form:', f.prefixstr())
        # print('infix form:', f)
        open_branches = a_tableux([f])
        # print('open branches:', open_branches)
        opbr = simplify(open_branches)
        print('minimized open branches:', opbr)
        if len(open_branches) == 0:
            print('UNSAT')
            continue
        print('SAT')

        if f.prefixstr() in sats:
            print('already generated')
            continue

        contras = close_open_branches(opbr)
        print('contras:', contras)
        neg_sel_br = neg_con(contras)
        closed_f = ConForm(f, neg_sel_br)
        print('closed formula:', closed_f.prefixstr())
        open_branches = a_tableux([closed_f])
        print('branches:', open_branches)
        if len(open_branches) > 0:
            print('alarm')
        sats.add(f.prefixstr())
        unstas.add(closed_f.prefixstr())

    return sats, unstas


def generate_data_file(filename, n_branches, n_vars, tests, trains, validation):
    sats, unsats = generate_dataset(alg=1, N=tests + trains + validation, bn=n_branches, vs=n_vars, s=0.2, ops=140,
                                    remove_dn=True)
    sats = list(sats)
    unsats = list(unsats)
    datasets = {}
    for lbl, sz in zip(['train', 'test', 'validation'], [trains, tests, validation]):
        datasets[lbl] = []
        print(lbl, sz)
        for _ in range(0, sz, 2):
            s = random.choice(sats)
            datasets[lbl].append((1, s))
            sats.remove(s)
            us = random.choice(unsats)
            datasets[lbl].append((0, us))
            unsats.remove(us)
    for lbl in ['train', 'test', 'validation']:
        with open(filename.format(lbl), 'w') as fl:
            for l, s in datasets[lbl]:
                fl.write('{} {}\n'.format(l, s))
    print('done')


generate_data_file("data.txt", 15,
                   10, 10000, 50000, 10000)
