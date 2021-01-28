
import torch
from torch.cuda.amp import autocast

from test_jit import JitTestCase
from torch.testing._internal.common_utils import run_tests


class TestAutocast(JitTestCase):
    def setUp(self):
        self.a_float32 = torch.rand((8, 8), device="cuda")
        self.b_float32 = torch.rand((8, 8), device="cuda")
        self.c_float32 = torch.rand((8, 8), device="cuda")
        self.d_float32 = torch.rand((8, 8), device="cuda")
        super().setUp()

    def tearDown(self):
        super().tearDown()

    def basic_helper(a, b):
        with autocast():
            e = torch.mm(a, b)
        return e

    def test_basic(self):
        jit_fn = torch.jit.script(self.basic_helper)
        result = jit_fn(self.a_float32, self.b_float32)
        self.assertEqual(result.dtype, torch.float16)


if __name__ == '__main__':
    run_tests()
