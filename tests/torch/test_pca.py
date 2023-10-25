from numpy.testing import assert_allclose
from sklearn.datasets import make_regression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn_utilities.torch.pca import PCATorch


def test_pca_implementation():
    # Load the data
    X, y = make_regression(n_samples=200, n_features=6, n_targets=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Fit PCA using sklearn
    pca_sklearn = PCA(n_components=2)
    pca_sklearn.fit(X_train)
    X_sklearn = pca_sklearn.transform(X_train)
    X_skleran_test = pca_sklearn.transform(X_test)

    # Fit PCA using PCATorch
    pca_torch = PCATorch(n_components=2)
    pca_torch.fit(X_train)
    X_torch = pca_torch.transform(X_train)
    X_torch_test = pca_torch.transform(X_test)

    # Compare the results
    assert_allclose(X_sklearn, X_torch, rtol=1e-4, atol=1e-4)
    assert_allclose(X_skleran_test, X_torch_test, rtol=1e-4, atol=1e-4)
