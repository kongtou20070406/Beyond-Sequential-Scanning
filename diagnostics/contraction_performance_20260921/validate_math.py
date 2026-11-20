"""Independent double-precision derivative check and exact scalar checks."""
import json
from pathlib import Path
import torch,sympy as s
P=Path(__file__).parent
torch.manual_seed(712);errors=[]
for magnitude in [.01,1.,100.]:
    x=(torch.randn(8,dtype=torch.float64)*magnitude).requires_grad_(True)
    alpha=torch.tensor(.7,dtype=torch.float64);eta=(1+alpha*x.square().sum()).rsqrt()
    analytic=eta*torch.eye(8,dtype=torch.float64)-alpha*eta**3*torch.outer(x,x)
    auto=torch.autograd.functional.jacobian(lambda y:y*(1+alpha*y.square().sum()).rsqrt(),x)
    errors.append(float((analytic-auto).abs().max().detach()))
h=s.symbols('h',real=True)
dynamic=s.Rational(95,100)*s.tanh(2*h)*(1+h/s.sqrt(1+h*h))
assert s.diff(dynamic,h).subs(h,0)==s.Rational(19,10)
# A q-contraction may return any wrong constant target: fixed point b/(1-q).
q,b=s.symbols('q b',real=True)
assert s.simplify(q*(b/(1-q))+b-b/(1-q))==0
# g=4 retention bound <= 1: exact polynomial gap.
t=s.symbols('t',nonnegative=True)
gap=s.expand(1-(1-t)*(1+t/2)**2)
assert gap==t**2*(t+3)/4 or s.simplify(gap-t**2*(t+3)/4)==0
# Constant-input scalar recurrence makes state-initialization decay coexist with input sensitivity.
K=5;u=s.symbols('u')
z=s.Integer(0)
for _ in range(K):z=q*z+u
assert s.simplify(s.diff(z,u)-sum(q**i for i in range(K)))==0
result=dict(torch=torch.__version__,sympy=s.__version__,max_derivative_error=max(errors),
            exact_dynamic_derivative_at_zero='19/10',group4_bound_gap=str(gap),
            input_sensitivity_K5=str(s.diff(z,u)),scope='Algebra and small derivative checks; no formal theorem verification.')
(P/'math_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
