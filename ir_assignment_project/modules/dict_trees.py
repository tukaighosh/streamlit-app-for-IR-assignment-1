"""
Dictionary Search structures (Lecture 4): Binary Search Tree and B-Tree,
each storing term -> postings list, plus a timing-experiment helper to
compare query/search time and retrieval time.
"""

import time
import random


# ---------------------------------------------------------------------------
# Binary Search Tree
# ---------------------------------------------------------------------------

class BSTNode:
    __slots__ = ("term", "postings", "left", "right")

    def __init__(self, term, postings):
        self.term = term
        self.postings = postings
        self.left = None
        self.right = None


class BST:
    def __init__(self):
        self.root = None
        self.size = 0

    def insert(self, term, postings):
        self.size += 1
        if self.root is None:
            self.root = BSTNode(term, postings)
            return
        node = self.root
        while True:
            if term == node.term:
                node.postings = postings
                return
            elif term < node.term:
                if node.left is None:
                    node.left = BSTNode(term, postings)
                    return
                node = node.left
            else:
                if node.right is None:
                    node.right = BSTNode(term, postings)
                    return
                node = node.right

    def search(self, term):
        """Returns postings list or None. Also returns nodes-visited (for depth info)."""
        node = self.root
        visited = 0
        while node is not None:
            visited += 1
            if term == node.term:
                return node.postings, visited
            elif term < node.term:
                node = node.left
            else:
                node = node.right
        return None, visited

    def prefix_search(self, prefix):
        """All terms starting with `prefix` (in-order traversal filter)."""
        results = []

        def inorder(node):
            if node is None:
                return
            inorder(node.left)
            if node.term.startswith(prefix):
                results.append(node.term)
            inorder(node.right)

        inorder(self.root)
        return results


def build_bst(dictionary: dict) -> BST:
    """dictionary: {term: postings_list}. Insert in shuffled order to avoid worst-case skew from sorted input."""
    tree = BST()
    items = list(dictionary.items())
    random.shuffle(items)
    for term, postings in items:
        tree.insert(term, postings)
    return tree


# ---------------------------------------------------------------------------
# B-Tree (simplified, order = MAX_KEYS + 1)
# ---------------------------------------------------------------------------

class BTreeNode:
    def __init__(self, leaf=True):
        self.leaf = leaf
        self.keys = []       # list of terms, sorted
        self.postings = []   # postings_list aligned with keys
        self.children = []   # child BTreeNode list (len = len(keys)+1)


class BTree:
    """A simplified in-memory B-Tree (minimum degree t, so each node holds
    up to 2t-1 keys and 2t children) supporting insert & search, matching
    the 'multiple children per node, balanced' structure described in
    Lecture 4."""

    def __init__(self, t=3):
        self.t = t  # minimum degree
        self.root = BTreeNode(leaf=True)
        self.size = 0

    def search(self, term, node=None):
        if node is None:
            node = self.root
        visited = 1
        i = 0
        while i < len(node.keys) and term > node.keys[i]:
            i += 1
        if i < len(node.keys) and term == node.keys[i]:
            return node.postings[i], visited
        if node.leaf:
            return None, visited
        result, sub_visited = self.search(term, node.children[i])
        return result, visited + sub_visited

    def insert(self, term, postings):
        self.size += 1
        root = self.root
        if len(root.keys) == (2 * self.t) - 1:
            new_root = BTreeNode(leaf=False)
            new_root.children.append(root)
            self._split_child(new_root, 0)
            self.root = new_root
            self._insert_non_full(new_root, term, postings)
        else:
            # if term already exists, just update
            if self._update_if_exists(root, term, postings):
                self.size -= 1
                return
            self._insert_non_full(root, term, postings)

    def _update_if_exists(self, node, term, postings):
        i = 0
        while i < len(node.keys) and term > node.keys[i]:
            i += 1
        if i < len(node.keys) and node.keys[i] == term:
            node.postings[i] = postings
            return True
        if node.leaf:
            return False
        return self._update_if_exists(node.children[i], term, postings)

    def _split_child(self, parent, i):
        t = self.t
        child = parent.children[i]
        new_node = BTreeNode(leaf=child.leaf)

        mid_key = child.keys[t - 1]
        mid_posting = child.postings[t - 1]

        new_node.keys = child.keys[t:]
        new_node.postings = child.postings[t:]
        child.keys = child.keys[:t - 1]
        child.postings = child.postings[:t - 1]

        if not child.leaf:
            new_node.children = child.children[t:]
            child.children = child.children[:t]

        parent.children.insert(i + 1, new_node)
        parent.keys.insert(i, mid_key)
        parent.postings.insert(i, mid_posting)

    def _insert_non_full(self, node, term, postings):
        i = len(node.keys) - 1
        if node.leaf:
            node.keys.append(None)
            node.postings.append(None)
            while i >= 0 and term < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                node.postings[i + 1] = node.postings[i]
                i -= 1
            node.keys[i + 1] = term
            node.postings[i + 1] = postings
        else:
            while i >= 0 and term < node.keys[i]:
                i -= 1
            i += 1
            if len(node.children[i].keys) == (2 * self.t) - 1:
                self._split_child(node, i)
                if term > node.keys[i]:
                    i += 1
            self._insert_non_full(node.children[i], term, postings)

    def all_terms(self):
        results = []

        def walk(node):
            if node.leaf:
                results.extend(node.keys)
                return
            for i in range(len(node.keys)):
                walk(node.children[i])
                results.append(node.keys[i])
            walk(node.children[-1])

        walk(self.root)
        return results


def build_btree(dictionary: dict, t=3) -> BTree:
    tree = BTree(t=t)
    for term, postings in dictionary.items():
        tree.insert(term, postings)
    return tree


# ---------------------------------------------------------------------------
# Timing experiment
# ---------------------------------------------------------------------------

def run_timing_experiment(dictionary: dict, query_terms: list, t=3):
    """
    Builds both structures and times `search` for each query term.
    Returns: rows (per-query timings) + summary (averages).
    """
    bst = build_bst(dictionary)
    btree = build_btree(dictionary, t=t)

    rows = []
    for term in query_terms:
        t0 = time.perf_counter()
        bst_result, bst_visited = bst.search(term)
        bst_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        btree_result, btree_visited = btree.search(term)
        btree_time = time.perf_counter() - t0

        rows.append({
            "term": term,
            "found_in_dict": term in dictionary,
            "bst_time_us": round(bst_time * 1e6, 3),
            "bst_nodes_visited": bst_visited,
            "btree_time_us": round(btree_time * 1e6, 3),
            "btree_nodes_visited": btree_visited,
        })

    avg_bst = round(sum(r["bst_time_us"] for r in rows) / len(rows), 3) if rows else 0
    avg_btree = round(sum(r["btree_time_us"] for r in rows) / len(rows), 3) if rows else 0

    summary = {
        "avg_bst_time_us": avg_bst,
        "avg_btree_time_us": avg_btree,
        "dictionary_size": len(dictionary),
        "faster_structure": "BST" if avg_bst < avg_btree else "B-Tree",
    }
    return rows, summary, bst, btree
