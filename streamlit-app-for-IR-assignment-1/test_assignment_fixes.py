import sys, site
sys.path.append(site.getusersitepackages())
import unittest
import os
from app import (
    BinarySearchTree, BTree, edit_distance, build_kgram_index, soundex
)

class TestIRAppFixes(unittest.TestCase):
    def test_btree_duplicate_insertion_and_search(self):
        bt = BTree(t=3)
        # Insert initial batch to trigger root split
        terms = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta']
        for t in terms:
            bt.insert(t, 'doc1')

        # Verify initial search
        found, docs = bt.search('gamma')
        self.assertTrue(found)
        self.assertEqual(docs, {'doc1'})

        # Re-insert existing key 'gamma' with new doc ID
        bt.insert('gamma', 'doc2')
        found, docs = bt.search('gamma')
        self.assertTrue(found)
        self.assertIn('doc1', docs)
        self.assertIn('doc2', docs)
        self.assertEqual(docs, {'doc1', 'doc2'})

    def test_iterative_bst(self):
        bst = BinarySearchTree()
        # Insert 1500 terms sequentially to test stack safety
        for i in range(1500):
            bst.insert(f"term_{i:04d}", f"doc_{(i%5)+1}")
        
        found, docs = bst.search("term_0750")
        self.assertTrue(found)
        self.assertEqual(docs, {'doc_1'})

    def test_edit_distance(self):
        self.assertEqual(edit_distance("kitten", "sitting"), 3)
        self.assertEqual(edit_distance("information", "information"), 0)
        self.assertEqual(edit_distance("retrieval", "retrival"), 1)

    def test_kgram_and_soundex(self):
        k_idx = build_kgram_index(["technology", "technical"], k=2)
        self.assertIn("technology", k_idx["$t"])
        self.assertIn("technical", k_idx["$t"])

        self.assertEqual(soundex("Robert"), soundex("Rupert"))
        self.assertEqual(soundex("Rubin"), "R150")

if __name__ == "__main__":
    unittest.main()
