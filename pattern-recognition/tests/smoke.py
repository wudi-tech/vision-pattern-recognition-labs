"""Small synthetic regression checks; no course data or accuracy reproduction."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')

import ast
import importlib.util
import json
from pathlib import Path
import platform
import tempfile
import unittest

import numpy as np
from PIL import Image
from scipy.stats import multivariate_normal

ROOT = Path(__file__).resolve().parents[1]

def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

bayes = load('bayes_boundary', 'work1/bayes_decision_boundary1.py')
nemo = load('pixel_bayes', 'work2/bayes_nemo.py')
gen = load('synthetic_generator', 'work3/task3_2_1v0.py')
svm = load('svm_utils', 'work3/test3_svm_utils.py')
lbp = load('lbp_only', 'work3/test3_LBP.py')
pixel_lbp = load('pixel_lbp', 'work3/test3_4.py')
hog_pipeline = load('hog_pipeline', 'work3/test3_5.py')
baseline = load('baseline', 'work3/test3_1.py')
plot_svm = load('plot_svm', 'work3/task3_2_2v1.py')
RESULTS = {}

class Smoke(unittest.TestCase):
    def test_gaussian_discriminant_reference_and_priors(self):
        X = np.array([[0., 0.], [1., 2.], [-2., 1.]])
        mu = np.array([.5, -.5]); cov = np.array([[2., .3], [.3, 1.]])
        actual = bayes.log_discriminant(X, mu, cov, .3)
        expected = multivariate_normal.logpdf(X, mu, cov) + np.log(.3) + np.log(2*np.pi)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        np.testing.assert_allclose(bayes.log_discriminant(X, mu, cov, .6)-actual, np.log(2))

    def test_pixel_classifiers_independent_queries(self):
        rng = np.random.default_rng(20261025)
        X = np.r_[rng.normal(.2, .025, (24,4)), rng.normal(.8,.025,(24,4))]
        positive=np.arange(48)<24; negative=~positive
        query=np.array([[.22,.19,.21,.2],[.81,.79,.8,.82]])
        for cls in [nemo.GaussianBayes,nemo.ParzenBayes,nemo.KNNBayes]:
            np.testing.assert_array_equal(cls().fit(X,positive,negative).predict(query),[1,-1])
        np.testing.assert_array_equal(nemo.GaussianBayes1D().fit(X,positive,negative).predict(query[:,0]),[1,-1])
        RESULTS['pixel_training_samples']=48

    def test_parzen_matches_direct_density_product(self):
        X=np.array([[.1,.2],[.2,.3],[.8,.7],[.9,.8]])
        pos=np.array([1,1,0,0],dtype=bool)
        model=nemo.ParzenBayes().fit(X,pos,~pos)
        Q=np.array([[.12,.22],[.75,.65],[.5,.4]])
        scores=[]
        for samples,h,prior in [(model.X1,model.h1,model.prior1),(model.X2,model.h2,model.prior2)]:
            density=np.exp(-.5*((Q[:,None,:]-samples[None,:,:])/h)**2)/(h*np.sqrt(2*np.pi))
            scores.append(density.mean(axis=1).prod(axis=1)*prior)
        np.testing.assert_array_equal(model.predict(Q),np.where(scores[0]>=scores[1],1,-1))

    def test_knn_training_size_boundary_tie_and_invalid_k(self):
        X=np.array([[0.],[1.]]); pos=np.array([True,False]); Q=np.array([[.1],[.9]])
        np.testing.assert_array_equal(nemo.KNNBayes(1).fit(X,pos,~pos).predict(Q),[1,-1])
        for k in [2,3]:
            pred=nemo.KNNBayes(k).fit(X,pos,~pos).predict(Q)
            np.testing.assert_array_equal(pred,[1,1])
            self.assertTrue(set(pred).issubset({-1,1}))
        for k in [0,-1,1.5]:
            with self.assertRaises(ValueError): nemo.KNNBayes(k)
        with self.assertRaises(ValueError):
            nemo.KNNBayes().fit(np.empty((0,1)),np.array([],dtype=bool),np.array([],dtype=bool)).predict(Q)

    def test_generated_image_pixel_feature_and_morphology_path(self):
        class RedClassifier:
            def predict(self, features):
                self.features=features
                return np.where(features[:,1]>.5,1,-1)
        image=np.zeros((48,48,3),dtype=np.uint8)
        image[12:36,12:36]=[230,110,30]
        model=RedClassifier()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'generated.png'; Image.fromarray(image).save(path)
            rgb,gray,result=nemo.apply_mask_and_classify(path,np.ones((24,24),dtype=bool),model)
        self.assertEqual(result.shape,(48,48))
        self.assertEqual(result.dtype,np.uint8)
        self.assertEqual(result[24,24],255)
        self.assertEqual(result[5,5],0)
        self.assertEqual(model.features.shape,(48*48,4))
        np.testing.assert_allclose(model.features[:,1:],rgb.reshape(-1,3)/255.)
        np.testing.assert_allclose(model.features[:,0],gray.ravel()/255.)

    def test_lbp_matches_pixelwise_oracle(self):
        images=np.random.default_rng(123).integers(0,256,(3,19,19)).astype(float)
        oracle=np.zeros((3,17,17),dtype=np.uint8)
        for n in range(3):
            for r in range(1,18):
                for c in range(1,18):
                    oracle[n,r-1,c-1]=sum(int(images[n,r+dr,c+dc]>=images[n,r,c])<<bit for bit,(dr,dc) in enumerate(lbp._NEIGHBOR_OFFSETS))
        np.testing.assert_array_equal(lbp._compute_lbp_codes(images),oracle)
        np.testing.assert_array_equal(pixel_lbp._compute_lbp_codes(images),oracle)
        lut=lbp._make_uniform_lut()
        self.assertEqual(int((lut!=58).sum()),58)
        self.assertEqual(set(lut),set(range(59)))

    def test_lbp_dimensions_normalization_and_uniform_brightness(self):
        X=np.random.default_rng(124).normal(size=(5,361))
        features=lbp.extract_lbp_features(X)
        self.assertEqual(features.shape,(5,531))
        np.testing.assert_allclose(features.reshape(5,9,59).sum(axis=2),1.)
        np.testing.assert_allclose(lbp.extract_lbp_features(X+50),features)
        self.assertEqual(pixel_lbp.build_features(X).shape,(5,597))
        constant=lbp._compute_lbp_codes(np.ones((1,19,19)))
        np.testing.assert_array_equal(constant,255)

    def test_hog_spatial_stats_and_combined_feature(self):
        X=np.random.default_rng(125).normal(size=(4,361))
        self.assertEqual(svm.spatial_stats(X).shape,(4,42))
        hog=svm.hog_features(X)
        self.assertEqual(hog.shape,(4,81))
        self.assertTrue(np.isfinite(hog).all())
        np.testing.assert_allclose((hog.reshape(4,9,9)**2).sum(axis=2),1.,atol=1e-6)
        np.testing.assert_array_equal(svm.hog_features(np.ones((1,361))),0.)
        self.assertEqual(hog_pipeline.build_features(X).shape,(4,484))
        np.testing.assert_allclose(hog_pipeline.build_features(X)[:,403:],hog)

    def test_scaler_uses_training_statistics_only(self):
        train=np.array([[0.,1.],[2.,1.]])
        for scaler_cls in [svm.MinMaxScaler,lbp.MinMaxScaler,baseline.MinMaxScaler]:
            scaler=scaler_cls().fit(train)
            np.testing.assert_allclose(scaler.transform(np.array([[4.,1.]])),[[3.,-1.]])
            np.testing.assert_allclose(scaler.xmin,[0.,1.])

    def test_actual_libsvm_heldout_circle_and_baseline_wrappers(self):
        X,y=gen.make_circle(30,seed=126)
        Q,truth=gen.make_circle(20,seed=127)
        Xs,Qs,_=svm.scale_train_test(X,Q)
        model=svm.train_final_rbf(y,Xs,C=10,gamma=2)
        score=float(np.mean(svm.predict(model,Qs)==truth))
        self.assertGreaterEqual(score,.95)
        model2=baseline.train_final(y,Xs,C=10,gamma=2)
        np.testing.assert_array_equal(baseline.predict(model2,Qs),svm.predict(model,Qs))
        np.testing.assert_allclose(plot_svm.libsvm_x_to_numpy(plot_svm.numpy_to_libsvm_x(X)),X)
        self.assertGreater(len(plot_svm.get_support_vectors(model)),0)
        RESULTS.update(circle_train=60,circle_heldout=40,circle_heldout_accuracy=score)

    def test_hog_to_libsvm_toy_patch_pipeline(self):
        rng=np.random.default_rng(128)
        labels=np.r_[np.ones(12),-np.ones(12)]
        xx,yy=np.meshgrid(np.arange(19),np.arange(19))
        images=np.array([xx if y==1 else yy for y in labels],dtype=float)
        images+=rng.normal(0,.03,images.shape)
        tr=np.r_[np.arange(8),np.arange(12,20)]; te=np.r_[np.arange(8,12),np.arange(20,24)]
        features=hog_pipeline.build_features(images.reshape(24,361))
        scaler=svm.MinMaxScaler().fit(features[tr])
        model=svm.train_final_rbf(labels[tr],scaler.transform(features[tr]),C=10,gamma=.01)
        accuracy=float(np.mean(svm.predict(model,scaler.transform(features[te]))==labels[te]))
        self.assertEqual(accuracy,1.)
        RESULTS.update(patch_train=16,patch_heldout=8,patch_heldout_accuracy=accuracy)

    def test_all_public_python_syntax(self):
        paths=list(ROOT.rglob('*.py'))
        for p in paths: ast.parse(p.read_text(encoding='utf-8-sig'))
        RESULTS['python_sources']=len(paths)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Smoke)
    run=unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps(dict(status='PASS' if run.wasSuccessful() else 'FAIL',tests=run.testsRun,
        scope='Synthetic CPU regression only; not face/Nemo benchmark reproduction',
        python=platform.python_version(),numpy=np.__version__,**RESULTS),ensure_ascii=False))
    raise SystemExit(0 if run.wasSuccessful() else 1)
