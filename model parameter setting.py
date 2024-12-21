random_state_new = 969

LR
LogisticRegression(
    penalty='l2',
    *,
    dual=False,
    tol=0.0001,
    C=1.0,
    fit_intercept=True,
    intercept_scaling=1,
    class_weight=None,
    random_state=None,
    solver='liblinear',
    max_iter=100,
    multi_class='auto',
    verbose=0,
    warm_start=False,
    n_jobs=None,
    l1_ratio=None,
)


AB 
AdaBoostClassifier(
    base_estimator=None, 
    *, 
    n_estimators=10, 
    learning_rate=1.0, 
    algorithm="SAMME.R", 
    random_state=random_state_new
)


MLP
MLPClassifier(
    hidden_layer_sizes=(100,),
    activation='relu',
    *,
    solver='lbfgs',
    alpha=0.0001,
    batch_size='auto',
    learning_rate='constant',
    learning_rate_init=0.001,
    power_t=0.5,
    max_iter=200,
    shuffle=True,
    random_state=None,
    tol=0.0001,
    verbose=False,
    warm_start=False,
    momentum=0.9,
    nesterovs_momentum=True,
    early_stopping=False,
    validation_fraction=0.1,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-08,
    n_iter_no_change=10,
    max_fun=15000,
)


BAG
BaggingClassifier(
    base_estimator=KNeighborsClassifier(), 
    n_estimators=10, 
    *, 
    max_samples=0.5, 
    max_features=0.5, 
    bootstrap=True, 
    bootstrap_features=False, 
    oob_score=False, 
    warm_start=False, 
    n_jobs=None,
    verbose=0, 
    random_state=random_state_new
)


GBM
GradientBoostingClassifier(
    *, 
    loss="log_loss", 
    learning_rate=0.1, 
    n_estimators=100, 
    subsample=1.0, 
    criterion="friedman_mse", 
    min_samples_split=2, 
    min_samples_leaf=1, 
    min_weight_fraction_leaf=0.0, 
    max_depth=3, 
    min_impurity_decrease=0.0, 
    init=None, 
    random_state=random_state_new, 
    max_features=None, 
    verbose=0, 
    max_leaf_nodes=None, 
    warm_start=False, 
    validation_fraction=0.1, 
    n_iter_no_change=None, 
    tol=1e-4, 
    ccp_alpha=0.0
)


XGB
XGBClassifier(
    base_score=0.5,
    booster='gbtree',
    colsample_bylevel=1,
    colsample_bynode=1,
    colsample_bytree=1,
    gamma=0,
    gpu_id=-1,
    importance_type='gain',
    interaction_constraints='',
    learning_rate=0.300000012,
    max_delta_step=0,
    max_depth=2,
    min_child_weight=1,
    missing=nan,
    monotone_constraints='()',
    n_estimators=100,
    n_jobs=8,
    num_parallel_tree=1,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1,
    subsample=1,
    tree_method='exact',
    validate_parameters=1,
    verbosity=None
)