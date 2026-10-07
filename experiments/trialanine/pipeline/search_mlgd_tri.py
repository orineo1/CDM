"""MLGD on the trialanine benchmark: experiments/synthetic/pipeline/search_mog.py (the shared vector-data
driver, all its flags) with the loss computed on the label-free count statistic (count_feature.py):
--count-meta <meta.json> wraps whatever --loss search_mog builds (kde-kl, mmd) as loss(c(pred), c(target))."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "..", "synthetic", "pipeline"))
import search_mog                      # noqa: E402
from count_feature import CountFeature, CountLoss   # noqa: E402

meta = sys.argv[sys.argv.index("--count-meta") + 1]
kappa = float(sys.argv[sys.argv.index("--count-kappa") + 1]) if "--count-kappa" in sys.argv else 8.0
for flag in ("--count-meta", "--count-kappa"):
    if flag in sys.argv:
        i = sys.argv.index(flag); del sys.argv[i:i + 2]
_orig = search_mog.loss_for_target
feat = CountFeature.from_meta(meta, kappa)
search_mog.loss_for_target = lambda args, gen, n_fit=10_000: CountLoss(_orig(args, gen, n_fit), feat)

if __name__ == "__main__":
    search_mog.main()
