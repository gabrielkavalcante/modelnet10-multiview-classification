from render_utils import run_cli

# 36 views per object, one every 10 degrees (0 to 350).
# Defaults reproduce the paper: 224x224 PNG, white background, gray object,
# orbital camera at 30 degrees elevation, distance 3.8, zoom 0.80.
#
#   python scripts/render_36views.py
#   python scripts/render_36views.py --elevation 10    # elevation ablation


if __name__ == "__main__":
    run_cli(n_views=36)
