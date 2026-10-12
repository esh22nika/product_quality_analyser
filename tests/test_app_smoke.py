from visionguard.ui.layout import create_app

def test_app_smoke():
    demo = create_app()
    assert demo is not None
    assert hasattr(demo, "launch")
