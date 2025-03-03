import numpy as np
from mvdr.mcca.mcca import MCCA, i_mcca, mcca_gevp

from .utils import (
    check_mcca_class,
    check_mcca_gevp,
    check_mcca_scores_and_loadings,
    generate_mcca_test_data,
    generate_mcca_test_settings,
)


def test_mcca():
    for Xs in generate_mcca_test_data():
        for params in generate_mcca_test_settings():
            n_views = len(Xs)

            # check basic usage of mcca_gevp
            out = mcca_gevp(Xs, **params)
            check_mcca_scores_and_loadings(
                Xs, out=out, regs=params["regs"], check_normalization=True
            )

            check_mcca_gevp(Xs=Xs, out=out, regs=params["regs"])

            # make sure centering went corrently
            for b in range(n_views):
                assert not out["centerers"][b].with_std
                if params["center"]:
                    assert np.allclose(out["centerers"][b].mean_, Xs[b].mean(axis=0))
                else:
                    assert out["centerers"][b].mean_ is None

            if params["regs"] is None:
                # check basic usage of i_mcca with SVD method
                out = i_mcca(Xs, signal_ranks=None, method="svd", **params)

                check_mcca_scores_and_loadings(
                    Xs, out=out, regs=params["regs"], check_normalization=True
                )

                check_mcca_gevp(Xs=Xs, out=out, regs=params["regs"])

            # check basic usage of i_mcca with gevp method
            # this solves GEVP by first doing SVD, not interesting in practice
            # but this should work correctly
            out = i_mcca(Xs, signal_ranks=None, method="gevp", **params)

            check_mcca_scores_and_loadings(
                Xs, out=out, regs=params["regs"], check_normalization=True
            )

            check_mcca_gevp(Xs=Xs, out=out, regs=params["regs"])

            # check i_mcca when we first do dimensionality reduction
            # with SVD method
            if params["regs"] is None:
                out = i_mcca(Xs, signal_ranks=[3] * n_views, method="svd", **params)

                check_mcca_scores_and_loadings(
                    Xs, out=out, regs=params["regs"], check_normalization=False
                )

            # check i_mcca when we first do dimensionality reduction
            # with GEVP method
            out = i_mcca(Xs, signal_ranks=[3] * n_views, method="gevp", **params)

            check_mcca_scores_and_loadings(
                Xs, out=out, regs=params["regs"], check_normalization=False
            )

            # check MCCA class
            mcca = MCCA(**params).fit(Xs)
            check_mcca_class(mcca, Xs)
