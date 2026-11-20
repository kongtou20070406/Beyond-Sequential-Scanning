"""Count model operations at the recorded input size without shadowing stdlib profile."""
import importlib.util
from supplement.runtime import ROOT


def profile(row, model, device):
    import torch
    from fvcore.nn import FlopCountAnalysis
    if row['mixer'] == 'spatial_mamba':
        if not device.startswith('cuda'):
            raise ValueError('The official scan profile requires CUDA.')
        model = model.to(device).eval()
        image = torch.zeros(1, 3, 256, 256, device=device)
        def scan(inputs, outputs):
            shape = inputs[0].type().sizes()
            b, d, length = shape
            state = inputs[2].type().sizes()[-1]
            return 7 * b * d * length * state
        def depthwise(inputs, outputs):
            b, c, h, w = inputs[0].type().sizes()
            return b * c * h * w * 121
        with torch.inference_mode():
            model(image)
        count = FlopCountAnalysis(model, image)
        count.set_op_handle('prim::PythonOp.SelectiveScanStateFn', scan,
                            'prim::PythonOp.DepthwiseFunction', depthwise)
        total = count.total()
        return {'input': [1, 3, 256, 256], 'parameters': sum(p.numel() for p in model.parameters()),
                'reported_macs': total, 'by_operator': dict(count.by_operator()),
                'unsupported_ops': dict(count.unsupported_ops()),
                'note': 'Official scan estimate and merged 11x11 depthwise convolution are included.'}
    spec = importlib.util.spec_from_file_location('paper_profile', ROOT / 'tools/profile_model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = model.to(device).eval()
    sample = torch.zeros(1, 3, 256, 256, device=device)
    with torch.inference_mode():
        extra, records = module.propagation_macs(model, sample)
        counter = FlopCountAnalysis(model, sample)
        counter.unsupported_ops_warnings(False).uncalled_modules_warnings(False).tracer_warnings('none')
        base = int(counter.total())
    result = {'input': [1, 3, 256, 256], 'parameters': sum(p.numel() for p in model.parameters()),
              'fvcore_macs': base, 'propagation_macs': extra, 'reported_macs': base + extra,
              'propagation_records': records, 'unsupported_ops': dict(counter.unsupported_ops())}
    if row.get('arm') == 'half_delay':
        result['extra_scalar_multiplications'] = 390070272
        result['extra_scalar_additions'] = 195035136
        result['note'] = 'Interpolation and history storage are additional costs outside the MAC convention.'
    return result
