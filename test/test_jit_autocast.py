
import torch
from torch.cuda.amp import autocast

from test_jit import JitTestCase
from torch.testing._internal.common_utils import run_tests


class TestAutocast(JitTestCase):
    def setUp(self):
        # common input tensors
        self.a_fp16 = torch.rand((8, 8), dtype=torch.float16, device='cuda')
        self.b_fp16 = torch.rand((8, 8), dtype=torch.float16, device='cuda')
        self.c_fp16 = torch.rand((8, 8), dtype=torch.float16, device='cuda')
        self.d_fp16 = torch.rand((8, 8), dtype=torch.float16, device='cuda')
        self.a_fp32 = torch.rand((8, 8), dtype=torch.float32, device='cuda')
        self.b_fp32 = torch.rand((8, 8), dtype=torch.float32, device='cuda')
        self.c_fp32 = torch.rand((8, 8), dtype=torch.float32, device='cuda')
        self.d_fp32 = torch.rand((8, 8), dtype=torch.float32, device='cuda')
        super().setUp()

    def tearDown(self):
        super().tearDown()

    def test_minimal(self):
        @torch.jit.script
        def fn(a, b):
            with autocast():
                return torch.mm(a, b)
        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float16)

    def test_minimal_cpu(self):
        @torch.jit.script
        def fn(a, b):
            with autocast():
                return torch.mm(a, b)
        result = fn(self.a_fp32.to('cpu'), self.b_fp32.to('cpu'))
        self.assertEqual(result.dtype, torch.float16)

    def test_minimal_off(self):
        @torch.jit.script
        def fn(a, b):
            with autocast(enabled=False):
                return torch.mm(a, b)
        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float32)

    def test_explicit_casts(self):
        @torch.jit.script
        def fn(a, b, c, d):
            with autocast():
                e = torch.mm(a.double(), b.double())
            f = torch.mm(c.double(), d.double())
            return e, f
        e, f = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        self.assertEqual(e.dtype, torch.float16)
        self.assertEqual(f.dtype, torch.float64)

    def test_duplicate_inputs(self):
        @torch.jit.script
        def fn(a, b, c, d):
            with autocast():
                e = torch.mm(a, a)
                f = torch.mm(e, e)
            return e, f
        e, f = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        self.assertEqual(e.dtype, torch.float16)
        self.assertEqual(f.dtype, torch.float16)

    def test_fp32_policy(self):
        @torch.jit.script
        def fn(a):
            with autocast(enabled=True):
                return torch.log(a)
        result = fn(self.a_fp16)
        self.assertEqual(result.dtype, torch.float32)

    def test_promote_policy(self):
        @torch.jit.script
        def fn(a, b, c, d):
            with autocast():
                e = torch.mm(a, b)
                f = torch.addcmul(e, c, d, value=0.1)
            return e, f
        e, f = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        self.assertEqual(e.dtype, torch.float16)
        self.assertEqual(f.dtype, torch.float32)

if __name__ == '__main__':
    run_tests()
