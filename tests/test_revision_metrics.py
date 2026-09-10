import numpy as np
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import roc_auc_score,average_precision_score,matthews_corrcoef,accuracy_score
from scripts.revision_metrics import EndpointMetrics,summarize


def test_weighted_binary_metrics_match_explicit_resampling_with_ties():
    y=np.array([0,1,1,0,0,1]);p=np.array([.1,.1,.7,.5,.7,.9]);w=np.array([2,3,1,0,2,1])
    yy=np.repeat(y,w);pp=np.repeat(p,w);result=EndpointMetrics(y,p,False)(w)
    for name,expected in [('auc_roc',roc_auc_score(yy,pp)),('auprc',average_precision_score(yy,pp)),
                          ('mcc',matthews_corrcoef(yy,pp>=.5)),('accuracy',accuracy_score(yy,pp>=.5))]:
        np.testing.assert_allclose(result[name],expected,atol=1e-12)


def test_weighted_correlations_match_resampled_ranks():
    y=np.array([1,2,2,3,4,5]);p=np.array([.1,.1,.7,.5,.7,.9]);w=np.array([2,3,1,0,2,1])
    yy=np.repeat(y,w);pp=np.repeat(p,w);result=EndpointMetrics(y,p,True)(w)
    np.testing.assert_allclose(result['pearson_r'],pearsonr(yy,pp).statistic,atol=1e-12)
    np.testing.assert_allclose(result['spearman_rho'],spearmanr(yy,pp).statistic,atol=1e-12)


def test_single_class_auc_is_undefined():
    result=EndpointMetrics(np.zeros(5),np.zeros(5),False)()
    assert np.isnan(result['auc_roc']) and np.isnan(result['sensitivity']) and np.isnan(result['auprc'])


def test_paired_aggregate_uses_common_molecules_and_raw_potency(tmp_path):
    y=np.tile(np.arange(12,dtype=float)[:,None],(1,8));y[:,[1,2,4,6,7]]%=2
    y[:3,3]=np.nan;y[-3:,5]=np.nan
    arms={}
    for name,offset in [('reference',.1),('candidate',.3)]:
        p=np.nan_to_num(y.copy());p[:,[0,3,5]]+=offset
        p[:,[1,2,4,6,7]]=.2+.6*p[:,[1,2,4,6,7]]
        arms[name]=tmp_path/name;arms[name].mkdir()
        np.savez(arms[name]/'ensemble_test_revision.npz',idx=np.arange(len(y)),labels=y,predictions=p)
    rows=summarize(arms,'synthetic','test','reference',bootstrap=30)
    row=next(r for r in rows if r['model']=='candidate' and r['metric']=='wrmse' and r['comparison']=='delta_vs_reference')
    np.testing.assert_allclose([row['estimate'],row['ci_lower'],row['ci_upper']],[.2,.2,.2],atol=1e-12)
