import argparse
from copy import deepcopy
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from experiments.e4_patchtst import run as old
from .train import fit, build
from .run import formal_gate


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.set_flush_denormal(True)
        torch.backends.mha.set_fastpath_enabled(False)
        torch.manual_seed(42)

    def test_all_models_start_exactly_at_e0_with_calibration(self):
        batch={"close":torch.randn(4,96),"market":torch.randn(4,96,7),"anchor":torch.randn(4,1),"alpha":.5}
        for kind in ('patchtst','linear_close','itransformer','linear_market'):
            config=dict(d_model=32,d_ff=64) if kind in ('patchtst','itransformer') else {}
            model=build({'kind':kind,'config':config}).eval()
            expected=batch['close'][:,-1:] if kind in ('patchtst','linear_close') else batch['anchor']
            torch.testing.assert_close(model(batch),expected.expand(-1,24),rtol=0,atol=0)

    def test_itransformer_receives_finite_nonzero_gradients(self):
        model=build({'kind':'itransformer','config':dict(d_model=32,d_ff=64)})
        batch={'market':torch.randn(4,96,7),'anchor':torch.randn(4,1)}
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        for _ in range(3):
            optimizer.zero_grad();error=model(batch)-torch.randn(4,24)
            (error.abs().mean()+.2*error.square().mean()).backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            optimizer.step()
        self.assertGreater(float(model.embedding.weight.grad.abs().sum()),0)

    def test_forward_alpha_matches_prior_head_folding(self):
        config=deepcopy(old.CONFIG);config.update(d_model=32,d_ff=64)
        parent=old.build_model('patchtst',config).eval()
        torch.nn.init.normal_(parent.head.weight,std=.01)
        torch.nn.init.normal_(parent.head.bias,std=.01)
        model=build({'kind':'patchtst','config':config}).eval()
        model.load_state_dict(parent.state_dict())
        x=torch.randn(4,96)
        with torch.no_grad():
            parent.head.weight.mul_(.5);parent.head.bias.mul_(.5)
            torch.testing.assert_close(model({'close':x,'alpha':.5}),parent({'close':x}),rtol=1e-5,atol=1e-6)

    def test_new_trainer_reproduces_original_raw_updates(self):
        cfg=deepcopy(old.CONFIG);cfg.update(d_model=32,d_ff=64)
        train=SimpleNamespace(x=torch.randn(32,96),y=torch.randn(32,24))
        dev=SimpleNamespace(x=torch.randn(16,96),y=torch.randn(16,24))
        class TinyAdapter:
            def __init__(self):
                self.train=SimpleNamespace(y=train.y.numpy());self.dev=SimpleNamespace(y=dev.y.numpy())
            def batch(self,samples,rows,mode):
                data=train if samples is self.train else dev
                return {'close':data.x[torch.from_numpy(rows)]}
            def anchor(self,samples):return dev.x[:,-1:].numpy()
        adapter=TinyAdapter()
        with TemporaryDirectory() as tmp:
            spec=importlib.util.spec_from_file_location('experiments.e4_patchtst._control_test',old.HERE/'run.py')
            parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
            parent.HERE=Path(tmp)/'old';parent.CONFIG=cfg
            parent.fit('patchtst',train,dev,argparse.Namespace(seed=42,resume=False,epochs=1,patience=4,batch_size=8))
            newhome=Path(tmp)/'new'
            fit({'id':'test','kind':'patchtst','mode':'close','config':cfg},42,adapter,newhome,
                dict(alpha=1.,max_epochs=1,patience=4,batch_size=8,min_delta=1e-5))
            a=torch.load(parent.HERE/'outputs/patchtst/training_state.pt',weights_only=True)['model']
            b=torch.load(newhome/'outputs/test/seed42/training_state.pt',weights_only=True)['model']
            for name,value in a.items():torch.testing.assert_close(value,b[name],rtol=0,atol=0)

    def test_formal_gate_rejects_mse_degradation_or_only_one_paired_win(self):
        score={'mae':.99,'mse':1.};ref={'mae':1.,'mse':1.};halves={'early':ref,'late':ref}
        self.assertTrue(formal_gate(score,ref,ref,2,halves,halves)['passed'])
        self.assertFalse(formal_gate(score,ref,ref,1,halves,halves)['passed'])
        score['mse']=1.0001
        self.assertFalse(formal_gate(score,ref,ref,2,halves,halves)['passed'])


if __name__=='__main__':unittest.main()
