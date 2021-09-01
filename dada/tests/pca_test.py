import torch
import unittest
import dada.tests.get_pca_no_interpolate as pca


"""
src:
+---+---+
| 0 | 1 |
+---+---+
| 2 | 3 |
+---+---+

src labels:
+---+---+
| 0 | 5 |
+---+---+
| 0 | 2 |
+---+---+

trg:
+---+---+
| a | b |
+---+---+
| c | d |
+---+---+

distance matrices:
+-----+-----+-----+-----+
| 0,a | 0,b | 0,c | 0,d |
+-----+-----+-----+-----+
| 1,a | 1,b | 1,c | 1,d |
+-----+-----+-----+-----+
| 2,a | 2,b | 2,c | 2,d |
+-----+-----+-----+-----+
| 3,a | 3,b | 3,c | 3,d |
+-----+-----+-----+-----+

+-----+-----+-----+-----+
| a,0 | a,1 | a,2 | a,3 |
+-----+-----+-----+-----+
| b,0 | b,1 | b,2 | b,3 |
+-----+-----+-----+-----+
| c,0 | c,1 | c,2 | c,3 |
+-----+-----+-----+-----+
| d,0 | d,1 | d,2 | d,3 |
+-----+-----+-----+-----+

"""


class TestSuite(unittest.TestCase):

      def test_get_pca(self):
        # Create two distance matrices for source and target
        source_distances = torch.zeros([4, 4])
        source_labels = torch.zeros([2, 2])
        target_distances = torch.zeros([4, 4])

        # Mark specific pixels for test
        source_distances[0,1] = 1
        source_distances[1,1] = 1

        source_labels[0,0] = 1
        source_labels[0,1] = 5
        source_labels[1,1] = 5

        target_distances[0,0] = 1
        target_distances[1,3] = 1
        
        result = pca.get_pixels_with_cycle_association(source_distances, target_distances, source_labels)
    
        self.assertEqual(result, [[1, 1, 3]])

if __name__ == '__main__':
    unittest.main(verbosity=2)
