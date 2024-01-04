
import os.path
import numpy.testing

from pypef.dca.gremlin_inference import GREMLIN


msa_file = os.path.abspath(
    os.path.join(
        os.path.abspath(__file__), 
        '../../datasets/AVGFP/uref100_avgfp_jhmmer_119.a2m'
    )
)


def test_gremlin():
    g = GREMLIN(
        alignment=msa_file,
        char_alphabet="ARNDCQEGHILKMFPSTWYV-",
        wt_seq=None,
        optimize=True,
        gap_cutoff=0.5,
        eff_cutoff=0.8,
        opt_iter=100
    )
    wt_score = g.get_wt_score()
    numpy.testing.assert_almost_equal(wt_score, 1203.549234202937, decimal=6)
    