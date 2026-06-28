import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_manifest():
    manifest_path = ROOT / "portfolio-web" / "demo-data" / "manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def test_demo_manifest_shape():
    data = load_manifest()

    assert data["project"] == "2D Animation LoRA Pipeline"
    assert data["mode"] == "cpu_stub_demo"
    assert data["summary"]["demo_ready"] is True
    assert data["summary"]["stages_total"] == 7
    assert len(data["stages"]) == 7
    assert data["product_results"]["metrics"]
    assert data["product_results"]["deliverables"]


def test_product_demo_assets_exist():
    data = load_manifest()
    assets = list(data["product_results"]["assets"].values())
    assets.extend(data["product_results"]["media"]["screenshots"])
    assets.append(data["product_results"]["media"]["video"])

    for relative_path in assets:
        asset_path = ROOT / "portfolio-web" / relative_path
        assert asset_path.exists(), relative_path
        assert asset_path.stat().st_size > 1000, relative_path


def test_portfolio_site_entrypoint_exists():
    html = (ROOT / "portfolio-web" / "index.html").read_text(encoding="utf-8")

    assert "2D Animation LoRA Pipeline" in html
    assert "demo-data/manifest.json" in html
    assert 'id="pipeline"' in html
    assert 'id="media"' in html
