"""Offline CPU check. Synthetic tensors only; no data/weights are downloaded."""
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Import the archived entry first: it retains its original Windows OpenMP setup.
import cifar10_multi as entry
import torch
from torch.utils.data import DataLoader, TensorDataset


def main():
    torch.set_num_threads(2)
    entry.set_seed(42)
    device = torch.device('cpu')
    rows = []
    with tempfile.TemporaryDirectory(prefix='cifar10-smoke-') as temp:
        for name in ('resnet18', 'resnet34', 'vgg16'):
            model = entry.build_model(name).to(device)
            model.eval()
            with torch.no_grad():
                output = model(torch.randn(2, 3, 32, 32))
            assert tuple(output.shape) == (2, 10)
            assert torch.isfinite(output).all()
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            first = next(model.parameters())
            before = first.detach().clone()
            # A single minibatch is exactly one optimizer step.
            loader = DataLoader(TensorDataset(torch.randn(2, 3, 32, 32), torch.tensor([0, 1])), batch_size=2)
            loss, _ = entry.train_one_epoch(model, loader, criterion, optimizer, device)
            assert math.isfinite(loss)
            assert not torch.equal(before, first.detach())
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            balanced = DataLoader(TensorDataset(torch.randn(10, 3, 32, 32), torch.arange(10)), batch_size=2)
            test_loss, _, confusion = entry.test(model, balanced, criterion, device, show_class_acc=True)
            assert math.isfinite(test_loss)
            assert [sum(row) for row in confusion] == [1] * 10
            entry.print_shape_table(model, name, 2, device)
            entry.save_confusion_matrix(confusion, name, temp)
            for suffix in ('', '_percent'):
                with (Path(temp)/f'{name}_confusion_matrix{suffix}.csv').open(encoding='utf-8', newline='') as f:
                    data = list(csv.reader(f))
                assert len(data) == 11 and all(len(row) == 11 for row in data)
            rows.append({'model': name, 'forward_shape': [2, 10], 'finite_loss': True,
                         'optimizer_steps': 1, 'parameters_changed': True, 'confusion_total': 10,
                         'parameters': sum(p.numel() for p in model.parameters())})
            del model, optimizer
        # Historical comparison models are import-safe; the early homework is AST-only.
        for filename, factory in [('resnet_cifar10_demo.py', 'resnet18_cifar10'),
                                  ('resnet34_cifar10_demo.py', 'resnet34_cifar10'),
                                  ('vgg16_cifar10_demo.py', 'VGG16CIFAR')]:
            spec = importlib.util.spec_from_file_location('legacy_' + filename[:-3], ROOT/'experiments'/filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            model = getattr(module, factory)().eval()
            with torch.no_grad():
                assert tuple(model(torch.zeros(2, 3, 32, 32)).shape) == (2, 10)
            print('HISTORICAL_FORWARD_PASS', filename)
            del model
    print('SYNTHETIC_CHECK_PASS ' + json.dumps({'torch': torch.__version__, 'torchvision': entry.torchvision.__version__,
          'device': 'cpu', 'data_downloaded': False, 'weights_downloaded': False, 'models': rows}))


if __name__ == '__main__':
    main()
