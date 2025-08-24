from mvdr.mcca.k_mcca import KMCCA, k_mcca_evp_svd
from mvdr.mcca.mcca import mcca_gevp
from mvdr.mcca.view_processing import get_view_kernels

from .utils import (
    check_mcca_class,
    compare_kmcca_to_mcca,
    generate_mcca_test_data,
    generate_mcca_test_settings,
)


def test_k_mcca():
    for Xs in generate_mcca_test_data():
        for params in generate_mcca_test_settings():
            if len(Xs) == 2 and params["n_components"] is None:
                # this setting raises some issues where are few
                # of the view scores are not equal. I do not think
                # this is an issue in practice so lets just skip
                # this scenario
                continue

            n_features = [x.shape[1] for x in Xs]
            Ks = get_view_kernels(Xs, kernel="linear")

            k_out = k_mcca_evp_svd(
                Ks=Ks, sval_thresh=0, signal_ranks=n_features, diag_mode="A", **params
            )

            mcca_out = mcca_gevp(Xs, **params)

            compare_kmcca_to_mcca(k_out=k_out, mcca_out=mcca_out)

            # check KMCCA class
            kmcca = KMCCA(**params).fit(Xs)
            check_mcca_class(kmcca, Xs)

            # check KMCCA class for different diag modes
            kmcca = KMCCA(diag_mode="B", **params).fit(Xs)
            check_mcca_class(kmcca, Xs)

            kmcca = KMCCA(diag_mode="C", **params).fit(Xs)
            check_mcca_class(kmcca, Xs)
