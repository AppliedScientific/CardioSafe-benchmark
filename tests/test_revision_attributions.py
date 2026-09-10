import numpy as np
import torch
from scripts.revision_attributions import integrated_gradients


def test_conditional_encoder_baseline_and_nonlinear_completeness():
    class Polynomial(torch.nn.Module):
        def forward(self,x):
            z=x[:,6164]
            return {'head':x[:,0]**3+2*x[:,1]*x[:,2]+z*x[:,3]+z**2}
    x=torch.zeros(2,6747);x[:,:4]=torch.tensor([[2.,3.,4.,5.],[-1.,2.,-3.,4.]])
    x[:,6164]=torch.tensor([7.,-2.])
    attr,gap,error=integrated_gradients(Polynomial(),x,'head',16)
    expected=np.zeros((2,6164));expected[:,0]=x[:,0].numpy()**3
    expected[:,1]=expected[:,2]=(x[:,1]*x[:,2]).numpy();expected[:,3]=(x[:,6164]*x[:,3]).numpy()
    np.testing.assert_allclose(attr,expected,atol=1e-5)
    np.testing.assert_allclose(attr.sum(1),gap,atol=1e-5);np.testing.assert_allclose(error,0,atol=1e-5)
