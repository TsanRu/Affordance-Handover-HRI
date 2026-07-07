import numpy as np
import unittest

import transformations_bk as tr
from ur_control.math_utils import (
    orientation_error_as_rotation_vector,
    quaternions_orientation_error,
    quaternion_normalize,
    quaternion_conjugate,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_slerp,
    quaternion_rotate_vector,
    diff_quaternion,
    random_quaternion,
    random_rotation_matrix,
    rotation_matrix_from_quaternion,
    quaternion_from_axis_angle,
    quaternion_from_ortho6,
    axis_angle_from_quaternion,
    ortho6_from_axis_angle,
    ortho6_from_quaternion,
    rotation_matrix_from_ortho6,
)


class TestQuaternionFunctions(unittest.TestCase):
    def setUp(self):
        self.quat_target = random_quaternion()
        self.quat_source = random_quaternion()
        self.axis_angle = axis_angle_from_quaternion(self.quat_source)
        self.ortho6 = ortho6_from_axis_angle(self.axis_angle)

    def test_orientation_error_as_rotation_vector(self):
        result = orientation_error_as_rotation_vector(self.quat_target, self.quat_source)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3,))

    def test_quaternions_orientation_error(self):
        result = quaternions_orientation_error(self.quat_target, self.quat_source)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (4,))

    def test_quaternion_normalize(self):
        q = np.array([1.0, 2.0, 3.0, 4.0])
        result = quaternion_normalize(q)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (4,))
        self.assertAlmostEqual(np.linalg.norm(result), 1.0)

    def test_quaternion_conjugate(self):
        result = quaternion_conjugate(self.quat_target)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (4,))
        self.assertTrue(np.allclose(tr.quaternion_conjugate(self.quat_target), result))

    def test_quaternion_inverse(self):
        result = quaternion_inverse(self.quat_target)
        self.assertIsInstance(result, np.ndarray)
        tr_result = tr.quaternion_inverse(self.quat_target)
        self.assertTrue(np.allclose(tr_result, result), msg=f"expected {tr_result} result {result}")

    def test_quaternion_multiply(self):
        result = quaternion_multiply(self.quat_target, self.quat_source)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (4,))
        tr_result = tr.quaternion_multiply(self.quat_target, self.quat_source)
        self.assertTrue(np.allclose(tr_result, result), msg=f"expected {tr_result} result {result}")

    def test_quaternion_slerp(self):
        result = quaternion_slerp(self.quat_target, self.quat_source, 0.5)
        self.assertIsInstance(result, np.ndarray)
        tr_result = tr.quaternion_slerp(self.quat_target, self.quat_source, fraction=0.5)
        self.assertTrue(np.allclose(tr_result, result), msg=f"expected {tr_result} result {result}")

    def test_quaternion_rotate_vector(self):
        vector = self.axis_angle
        result = quaternion_rotate_vector(self.quat_target, vector)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3,))
        tr_result = tr.quaternion_rotate_vector(self.quat_target, vector)
        self.assertTrue(np.allclose(tr_result, result), msg=f"expected {tr_result} result {result}")

    def test_diff_quaternion(self):
        result = diff_quaternion(self.quat_target, self.quat_source)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (4,))
        tr_result = tr.diff_quaternion(self.quat_target, self.quat_source)
        self.assertTrue(np.allclose(tr_result, result), msg=f"expected {tr_result} result {result}")

    def test_random_quaternion(self):
        result = random_quaternion()
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (4,))

    def test_random_rotation_matrix(self):
        result = random_rotation_matrix()
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (4, 4))

    def test_rotation_matrix_from_quaternion(self):
        q = random_quaternion()
        result = rotation_matrix_from_quaternion(q)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (4, 4))
        tr_result = tr.quaternion_matrix(q)
        self.assertTrue(np.allclose(tr_result, result), msg=f"expected {tr_result} result {result}")

    def test_quaternion_from_axis_angle(self):
        result = quaternion_from_axis_angle(self.axis_angle)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (4,))
        tr_result = tr.quaternion_from_axis_angle(self.axis_angle)
        self.assertTrue(np.allclose(tr_result, result), msg=f"expected {tr_result} result {result}")

    def test_quaternion_from_ortho6(self):
        result = quaternion_from_ortho6(self.ortho6)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (4,))
        tr_result = tr.quaternion_from_ortho6(self.ortho6)
        # ABS because the quaternion may be flipped
        self.assertTrue(np.allclose(np.abs(tr_result), np.abs(result)), msg=f"expected {tr_result} result {result}")

    def test_axis_angle_from_quaternion(self):
        result = axis_angle_from_quaternion(self.quat_target)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3,))
        tr_result = tr.axis_angle_from_quaternion(self.quat_target)
        self.assertTrue(np.allclose(tr_result, result, atol=1e-4), msg=f"expected {tr_result} result {result}")

    def test_ortho6_from_axis_angle(self):
        result = ortho6_from_axis_angle(self.axis_angle)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (6,))
        tr_result = tr.ortho6_from_axis_angle(self.axis_angle)
        self.assertTrue(np.allclose(tr_result, result), msg=f"expected {tr_result} result {result}")

    def test_ortho6_from_quaternion(self):
        result = ortho6_from_quaternion(self.quat_target)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (6,))
        tr_result = tr.ortho6_from_quaternion(self.quat_target)
        self.assertTrue(np.allclose(tr_result, result), msg=f"expected {tr_result} result {result}")

    def test_rotation_matrix_from_ortho6(self):
        result = rotation_matrix_from_ortho6(self.ortho6)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (4, 4))
        tr_result = tr.rotation_matrix_from_ortho6(self.ortho6)
        self.assertTrue(np.allclose(tr_result, result), msg=f"expected {tr_result} result {result}")


if __name__ == '__main__':
    unittest.main()
