import torch
import unittest
import dada.tests.get_pca_no_interpolate as pca



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
