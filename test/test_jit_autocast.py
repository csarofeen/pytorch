
import torch
from torch.cuda.amp import autocast

import unittest
from test_jit import JitTestCase
from torch.testing._internal.common_utils import run_tests


class TestAutocast(JitTestCase):
    def setUp(self):
        # common input tensors
        self.a_fp16 = torch.rand((2, 2), dtype=torch.float16, device='cuda')
        self.b_fp16 = torch.rand((2, 2), dtype=torch.float16, device='cuda')
        self.c_fp16 = torch.rand((2, 2), dtype=torch.float16, device='cuda')
        self.d_fp16 = torch.rand((2, 2), dtype=torch.float16, device='cuda')
        self.a_fp32 = torch.rand((2, 2), dtype=torch.float32, device='cuda')
        self.b_fp32 = torch.rand((2, 2), dtype=torch.float32, device='cuda')
        self.c_fp32 = torch.rand((2, 2), dtype=torch.float32, device='cuda')
        self.d_fp32 = torch.rand((2, 2), dtype=torch.float32, device='cuda')
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

    def test_runtime_autocast_state(self):
        @torch.jit.script
        def fn(a, b, use_amp: bool):
            with autocast(enabled=use_amp):
                return torch.mm(a, b)
        # runtime values for autocast enable argument are not supported
        with self.assertRaises(RuntimeError):
            fn(self.a_fp32, self.b_fp32, True)

    def test_runtime_autocast_state_expr(self):
        @torch.jit.script
        def fn(a, b):
            with autocast(enabled=True if a[0][0] > 0.5 else False):
                return torch.mm(a, b)
        # runtime values for autocast enable argument are not supported
        with self.assertRaises(RuntimeError):
            fn(self.a_fp32, self.b_fp32)

    def test_explicit_casts(self):
        @torch.jit.script
        def fn(a, b, c, d):
            with autocast():
                e = torch.mm(a.double(), b.double()).float()
                f = torch.mm(c, d).double()
            g = torch.mm(c.double(), f)
            return e, f, g
        e, f, g = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        self.assertEqual(e.dtype, torch.float32)
        self.assertEqual(f.dtype, torch.float64)
        self.assertEqual(g.dtype, torch.float64)

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

    # TODO: fix and enable this test
    @unittest.skipIf(True, "promote policy is currently broken")
    def test_promote_policy_fp64(self):
        @torch.jit.script
        def fn(a, b):
            with autocast(enabled=True):
                return torch.addcmul(a, a, b, value=0.1)
        result = fn(self.a_fp32.double(), self.b_fp32.double())
        self.assertEqual(result.dtype, torch.float64)

    def test_control_flow(self):
        @torch.jit.script
        def fn(a, b, c, d):
            with autocast():
                if a[0][0] > 0.5:
                    e = torch.mm(a, b)
                    x = 1
                else:
                    e = torch.mm(c, d)
                    x = 2
                f = torch.mm(d, e) * x
            return e, f
        e, f = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        self.assertEqual(e.dtype, torch.float16)
        self.assertEqual(f.dtype, torch.float16)

    # this works find in regular Python, but it creates a delicate
    # situation in TorchScript where the types are not consistent across
    # the then/else branches
    def test_divergent_types(self):
        @torch.jit.script
        def fn(a, b, c, d):
            with autocast():
                if a[0][0] > 0.5:
                    e = torch.mm(a, b)
                    f = torch.mm(a, b).float()
                else:
                    e = torch.mm(c, d).float()
                    f = torch.mm(a, b)
            return torch.mm(e.float(), f.float())
        result = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        self.assertEqual(result.dtype, torch.float32)

    # another, more complex case of divergent types
    def test_divergent_autocast(self):
        @torch.jit.script
        def fn(a, b, c, d):
            autocast_on = autocast(enabled=True)
            autocast_off = autocast(enabled=False)
            if a[0][0] > 0.5:
                with autocast_on:
                    e = torch.mm(a, b)
            else:
                with autocast_off:
                    e = torch.mm(c, d)
            return torch.mm(e, e)
        fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)

    def test_conditional_autocast(self):
        @torch.jit.script
        def fn(a, b):
            autocast_on = autocast(enabled=True)
            autocast_off = autocast(enabled=False)
            with autocast_on if a[0][0] > 0.5 else autocast_off:
                return torch.mm(a, b)
        # conditional autocast expressions are not supported
        with self.assertRaises(RuntimeError):
            fn(self.a_fp32, self.b_fp32)

    def test_nested_autocast(self):
        @torch.jit.script
        def fn(a, b, c, d):
            with autocast(enabled=False):
                e = torch.mm(a, b)
                with autocast(enabled=True):
                    f = torch.mm(e, c)
                    with autocast(enabled=False):
                        g = torch.mm(e, d)
            return e, f, g
        e, f, g = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        self.assertEqual(e.dtype, torch.float32)
        self.assertEqual(f.dtype, torch.float16)
        self.assertEqual(g.dtype, torch.float32)

    def test_reused_autocast(self):
        @torch.jit.script
        def fn(a, b, c, d):
            autocast_instance = autocast(enabled=True)
            with autocast_instance:
                e = torch.mm(a, b)
                with autocast_instance:
                    e = torch.mm(c, d)
                    f = torch.mm(d, e)
            g = torch.mm(e, f)
            return e, f, g
        e, f, g = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        self.assertEqual(e.dtype, torch.float16)
        self.assertEqual(f.dtype, torch.float16)
        self.assertEqual(g.dtype, torch.float16)

    def test_callees(self):
        def helper(a, b):
            return torch.mm(a, b)

        @torch.jit.script
        def fn(a, b):
            with autocast(enabled=True):
                tmp = helper(a, b)
                tmp = helper(tmp, tmp)
                tmp = helper(tmp, tmp)
                tmp = helper(tmp, tmp)
                return helper(tmp, b)
        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float16)


if __name__ == '__main__':
    run_tests()
