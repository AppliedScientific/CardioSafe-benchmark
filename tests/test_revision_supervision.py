import numpy as np
from train.revision_supervision import select_supervision


def test_herg_volume_is_nested_and_preserves_other_labels_and_holdout():
    y=np.arange(160,dtype=float).reshape(20,8);train=np.arange(15)
    small,small_idx,_,_=select_supervision(y,train,herg_fraction=.25)
    large,_,_,_=select_supervision(y,train,herg_fraction=.5)
    np.testing.assert_array_equal(small[:,3:],y[:,3:])
    np.testing.assert_array_equal(small[15:],y[15:])
    np.testing.assert_array_equal(small_idx,train)
    assert set(np.where(np.isfinite(small[:15,0]))[0])<=set(np.where(np.isfinite(large[:15,0]))[0])


def test_label_sweep_keeps_input_rows_when_all_their_labels_are_masked():
    y=np.full((20,8),np.nan);y[:,0:3]=1
    out,rows,_,_=select_supervision(y,np.arange(20),herg_fraction=.1)
    assert np.isfinite(out[:,0]).sum()==2
    np.testing.assert_array_equal(rows,np.arange(20))


def test_originally_unlabelled_rows_stay_excluded_at_every_fraction():
    y=np.full((20,8),np.nan);y[:18,:3]=1
    for fraction in [.1,.25,.5,1.]:
        _,rows,_,_=select_supervision(y,np.arange(20),herg_fraction=fraction)
        np.testing.assert_array_equal(rows,np.arange(18))


def test_single_channel_removes_other_supervision_and_unlabelled_training_rows():
    y=np.ones((10,8));y[0,5:7]=np.nan
    out,idx,active,record=select_supervision(y,np.arange(8),channel='cav12')
    assert active==(5,6) and 0 not in idx and len(idx)==7
    assert np.isnan(out[:,[0,1,2,3,4,7]]).all()
    np.testing.assert_array_equal(out[:,5:7],y[:,5:7])
