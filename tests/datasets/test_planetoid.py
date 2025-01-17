import unittest
from torch_geometric.datasets import Planetoid
from mmgraph.registry import DATASETS

class TestPlanetoid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.planetoid_instance = Planetoid(name='Cora', root='/tmp/Cora')

    def test_planetoid_registration(self):
        self.assertIn('Planetoid', DATASETS.module_dict)

    def test_planetoid_attributes(self):
        expected_attributes = ['x', 'edge_index', 'y']
        for attr in expected_attributes:
            self.assertTrue(hasattr(self.planetoid_instance, attr))

    def test_planetoid_data_shape(self):
        self.assertGreater(self.planetoid_instance[0].num_nodes, 0)
        self.assertGreater(self.planetoid_instance[0].num_edges, 0)
        self.assertGreater(self.planetoid_instance[0].num_features, 0)

    def test_planetoid_different_datasets(self):
        cora = Planetoid(name='Cora', root='/tmp/Cora')
        citeseer = Planetoid(name='CiteSeer', root='/tmp/CiteSeer')
        pubmed = Planetoid(name='PubMed', root='/tmp/PubMed')

        self.assertTrue(cora.num_classes != citeseer.num_classes or 
                        cora.num_classes != pubmed.num_classes)
