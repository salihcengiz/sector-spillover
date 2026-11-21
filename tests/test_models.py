"""Model ve eğitim testleri (torch gerekir; torch yoksa atlanır — yerel makinede torch kurulu değil,
bu testler Colab'da `pytest -q` ile çalışır).

Kontroller: çıktı boyutları, OwnSectorLSTM'de sektörler arası etki olmaması, eğitimin en iyi val
ağırlıklarını geri yüklemesi ve tahminlerin log-RV ölçeğine geri çevrilmesi.
"""
import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from spillover.data import DailyView, MonthlyView
from spillover.prep import SplitSpec, prepare

N, F, L = 5, 4, 12


def make_data(view=None, n_days=700, n_sectors=N, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2008-01-02", periods=n_days)
    returns = pd.DataFrame(rng.normal(0, 1.2, size=(n_days, n_sectors)).round(2), index=dates,
                           columns=[f"S{i}" for i in range(n_sectors)])
    panel = (view or DailyView()).build(returns)
    return prepare(panel, SplitSpec.sequential("2009-06-30", "2009-12-31", embargo=5), lookback=L)


def test_output_shapes_and_registry():
    if torch is None:
        return
    from spillover.models import MODELS, build_model
    assert {"cross_lstm", "own_lstm"} <= set(MODELS)
    x = torch.randn(3, L, N, F)
    for name in ("cross_lstm", "own_lstm"):
        for layers in (1, 2):
            model = build_model(name, N, F, hidden=8, layers=layers)
            assert model(x).shape == (3, N), name
    try:
        build_model("transformer", N, F)
    except ValueError:
        return
    raise AssertionError("bilinmeyen model tipi hata vermeliydi")


def test_own_sector_model_has_no_cross_effects():
    """Senaryo analizindeki kontrolün model tarafı: sektör 0'ın girdisini değiştirmek yalnızca sektör 0'ı etkiler."""
    if torch is None:
        return
    from spillover.models import build_model
    torch.manual_seed(0)
    x = torch.randn(4, L, N, F)
    shocked = x.clone()
    shocked[:, -1, 0, :] -= 3.0
    own, cross = build_model("own_lstm", N, F, hidden=8).eval(), build_model("cross_lstm", N, F, hidden=8).eval()
    with torch.no_grad():
        d_own = own(shocked) - own(x)
        d_cross = cross(shocked) - cross(x)
    assert d_own[:, 1:].abs().max() < 1e-6, "köşegen dışı etki 0 olmalı (yalnızca kayan nokta gürültüsü)"
    assert d_own[:, 0].abs().max() > 0
    assert d_cross[:, 1:].abs().max() > 0, "çapraz model diğer sektörlerden etkilenebilmeli"


def test_training_restores_best_weights_and_predicts():
    if torch is None:
        return
    from spillover.models import build_model
    from spillover.training import TrainConfig, dataset_loss, predict_panel, set_seed, train_model
    data = make_data()
    set_seed(0)
    model = build_model("cross_lstm", data.panel.N, data.panel.F, hidden=8)
    result = train_model(model, data, TrainConfig(epochs=4, batch_size=32, patience=2, device="cpu"))
    assert 1 <= result.best_epoch <= len(result.history) <= 4
    best = min(h["val_loss"] for h in result.history)
    assert np.isclose(result.best_val_loss, best)
    assert np.isclose(dataset_loss(model, data.dataset("val")), best, rtol=1e-5), "en iyi ağırlıklar geri yüklenmeli"

    test = data.splits.test
    pred = predict_panel(model, data, test)
    assert pred.shape == data.panel.y.shape
    assert np.isfinite(pred[test]).all() and np.isnan(np.delete(pred, test, axis=0)).all()
    with torch.no_grad():
        windows = data.X[test[:2, None] - L + 1 + np.arange(L)]
        manual = data.scaler.inverse_y(model.eval()(torch.from_numpy(windows)).numpy())
    assert np.allclose(pred[test[:2]], manual, atol=1e-5)


def test_same_seed_same_result_on_cpu():
    if torch is None:
        return
    from spillover.models import build_model
    from spillover.training import TrainConfig, set_seed, train_model
    data = make_data(MonthlyView(), n_days=2500)
    losses = []
    for _ in range(2):
        set_seed(7)
        model = build_model("own_lstm", data.panel.N, data.panel.F, hidden=4, embed_dim=2)
        res = train_model(model, data, TrainConfig(epochs=2, batch_size=16, device="cpu", seed=7))
        losses.append([h["val_loss"] for h in res.history])
    assert np.allclose(losses[0], losses[1])
