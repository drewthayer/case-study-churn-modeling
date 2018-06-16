import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# function imports
from data import *
from plotroc import plot_roc_curve
from bagging import bagging
from knn import knn
from random_forest import random_forest
# imports
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence

#%%
if __name__ == '__main__':
    df_train = pd.read_csv('../data/churn_train.csv')
    X_train, y_train = get_data(df_train)
    df_test = pd.read_csv('../data/churn_test.csv')
    X_test, y_test = get_data(df_test)

    # logistic regression
    pipe_logistic = Pipeline([
            ('scaler', StandardScaler()),
            ('logistic', LogisticRegression())
            ])
    pipe_logistic.fit(X_train, y_train)

    y_hat = pipe_logistic.predict(X_test)
    score_accuracy = accuracy_score(y_test, y_hat)
    score_precision = precision_score(y_test, y_hat)
    print('accuracy = {}'.format(score_accuracy))
    print('precision = {}'.format(score_precision))

    coefs = pipe_logistic.named_steps['logistic'].coef_

    print(coefs)

    # knn classifier
    knn = knn(X_train, X_test, y_train, y_test)

    # decision tree classifier
    dtc = DecisionTreeClassifier(max_depth=3,)
    dtc.fit(X_train, y_train);

    y_preds_dtc = dtc.predict_proba(X_test)[:,1]
    y_preds_dtc_bin = dtc.predict(X_test)

    dtc_prec = np.mean(cross_val_score(dtc, X_train, y_train, scoring='precision', cv=5))
    dtc_acc = np.mean(cross_val_score(dtc, X_train, y_train, scoring='accuracy', cv=5))
    dtc_test_prec = precision_score(y_test, y_preds_dtc_bin)
    dtc_test_acc = accuracy_score(y_test, y_preds_dtc_bin)
    print("DTC cross validated precision score is {:0.3}".format(dtc_prec))
    print("DTC cross validated accuracy score is {:0.3}".format(dtc_acc))
    print("DTC test precision score is {:0.3}".format(dtc_test_prec))
    print("DTC test accuracy score is {:0.3}".format(dtc_test_acc))

    # bagging classifier
    bagc = bagging(X_train, X_test, y_train, y_test)

    # random forest classifier
    rfc = random_forest(X_train, X_test,y_train, y_test)

    # adaboost classifier
    ada = AdaBoostClassifier()
    ada.fit(X_train, y_train)
    predict = ada.predict(X_test)
    print('AdaBoost accuracy:', ada.score(X_test, y_test))
    acc = cross_val_score(ada,X_test, y_test, cv=5, scoring='accuracy')
    precision = cross_val_score(ada,X_test, y_test, cv=5, scoring='precision')
    print('accuracy = {}'.format(np.mean(acc)))
    print('precision = {}'.format(np.mean(precision)))

    # gradient boost classifier
    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, y_train)
    y_hat = gbc.predict(X_test)
    score_accuracy = accuracy_score(y_test, y_hat)
    score_precision = precision_score(y_test, y_hat)
    print('accuracy = {}'.format(score_accuracy))
    print('precision = {}'.format(score_precision))

#%%
    #gradient boost grid search
    '''grid search to find best params for gradient boost classifier '''
    gbc_grid = {'learning_rate': np.linspace(0.2,0.8,4),
                      'max_depth': [1,2,4,8],
                      'min_samples_leaf': [2, 4, 6, 8],
                      'max_features': ['sqrt', 'log2', None],
                      'n_estimators': [100, 150, 200, 250, 300]}
    
    gbc_gridsearch = GridSearchCV(GradientBoostingClassifier(),
                         gbc_grid,
                         n_jobs=-1,
                         verbose=True,
                         scoring='precision')
    gbc_gridsearch.fit(X_train, y_train)
    
    
#%% gradient boost best model
    print("\nbest gbc parameters:", gbc_gridsearch.best_params_)  
    print("\ntraining cross-val precision score: {0:0.3f}".format(gbc_gridsearch.best_score_))
    print("number of cross-val folds {}".format(gbc_gridsearch.n_splits_))
    results = gbc_gridsearch.cv_results_
    gbc_best = gbc_gridsearch.best_estimator_
    
    y_hat = gbc_best.predict(X_test)
    score_accuracy = accuracy_score(y_test, y_hat)
    score_precision = precision_score(y_test, y_hat)
    print('\nbest gbc accuracy = {0:0.3f}'.format(score_accuracy))
    print('best gbc precision = {0:0.3f}'.format(score_precision))
#%%
     # roc curve for logistic
    y_test_preds = pipe_logistic.named_steps['logistic'].predict_proba(X_test)[:,1]
    plot_roc_curve(y_test,y_test_preds)
    
     # roc curve for adaboost
    y_test_preds = ada.predict_proba(X_test)[:,1]
    plot_roc_curve(y_test,y_test_preds)
#%%
    # plot muliple models on roc curve
    logistic_mod = pipe_logistic.named_steps['logistic']
    models = [logistic_mod, knn, dtc, bagc, rfc, ada, gbc]
    model_names = ['logistic', 'knn', 'decision tree', 'bagging', 'random forest', 'AdaBoost', 'Gradient Boost']
    colors = ['b','k','m','g','r','c','y']
    plt.figure()
    for idx, model in enumerate(models):
        modname = model_names[idx]
        color = colors[idx]
        y_test_preds = model.predict_proba(X_test)[:,1]
        plot_roc_curve(y_test,y_test_preds,modname,color)
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
#%%
    # gradient boost partial dependency plots
    features = [0,1,2,3,4,5]
    names = X_train.columns
    fig, axs = plot_partial_dependence(gbc, X_train, features,
                                       feature_names=names,
                                       n_jobs=-1, grid_resolution=50)
    fig.suptitle('Partial dependence plots')
    plt.subplots_adjust(top=0.9)  # tight_layout causes overlap with suptitle
    
    # save fig
    figname = 'partial_d_compare_rf'
    fig.set_size_inches(8, 5)
    plt.savefig('{}.png'.format(figname),format='png', dpi=300)

    #%%
    # plot feature importances
    impt_gbc = gbc.feature_importances_
    impt_gbc = impt_gbc/np.max(impt_gbc)
    #imp_gbc = impt_gbc.reshape(-1,1)
    impt_rfc = rfc.feature_importances_
    impt_rfc = impt_rfc/np.max(impt_rfc)
    #imp_rfc = impt_rfc.reshape(-1,1)
    names = np.array(X_train.columns)
    #names = names.reshape(-1,1)

    
    #imp_df = pd.DataFrame([names,imp_gbc,imp_rfc])
    
    #%% feature importances bar plot NOT WORKING
    fig, ax1 = plt.subplots(1,1,figsize=(18,3))
    ax1.set_title('Feature Importance from Gradient Boost (blue) and Random Forest(green)')
    ax1.bar(x=names, height=impt_gbc, alpha=0.2, color='b')
    ax1.set_xticklabels(X_train.columns, rotation=40);
    
    ax1.bar(x=names, height=impt_rfc, alpha=0.3, color='g')
    
    plt.show()
    # save fig
    figname = 'importances_bar'
    fig.set_size_inches(6, 4)
    plt.savefig('{}.png'.format(figname),format='png', dpi=300)
    
    #%% feature importances sorted bar plot
    names = np.array(X_train.columns)
    ft_imp_gbc = 100*gbc.feature_importances_ / np.sum(gbc.feature_importances_) # funny cause they sum to 1
    ft_imp_gbc_srt, ft_names_gbc, ft_idxs_gbc = zip(*sorted(zip(ft_imp_gbc, names, range(len(names)))))
    
    ft_imp_rfc = 100*rfc.feature_importances_ / np.sum(rfc.feature_importances_) # funny cause they sum to 1
    ft_imp_rfc_srt, ft_names_rfc, ft_idxs_gbc = zip(*sorted(zip(ft_imp_rfc, names, range(len(names)))))
    
    width = 0.8
    
    # combined plot is not quite right 
    
    idx = np.arange(len(names))
    #plt.barh(idx, ft_imp_gbc_srt, align='center', color='b',alpha=0.2)
    plt.barh(idx, ft_imp_rfc_srt, align='center', color='g',alpha=0.2)
    plt.yticks(idx, ft_names_rfc)
    
    plt.title("Feature Importances in Random Forest")
    plt.xlabel('Relative Importance of Feature', fontsize=14)
    plt.ylabel('Feature Name', fontsize=14)
    #plt.legend(['Gradient Booster','Random Forest'])
    
    figname = '../figs/feature-importances-rfc.png'
    plt.savefig(figname, bbox_inches='tight', dpi=300)
    
    #%% feature importances from logistic regression
    coefs_exp = np.exp(coefs[0])
    coefs_log_srt, ft_names_log, coef_idxs_log = zip(*sorted(zip(coefs_exp, names, range (len(names)))))
    
    idx = np.arange(len(names))
    #plt.barh(idx, ft_imp_gbc_srt, align='center', color='b',alpha=0.2)
    plt.barh(idx, coefs_log_srt, align='center', color='c',alpha=0.2)
    plt.yticks(idx, ft_names_log)
    
    plt.title('Exponentiated Feature Coefficients in Logistic Regression')
    
    figname = '../figs/coefficients-logistic-exp.png'
    plt.savefig(figname, bbox_inches='tight', dpi=300)
    